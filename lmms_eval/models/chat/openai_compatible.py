import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List


def _video_id_from_doc_path(path) -> str:
    """Extract video_id from a doc field that holds a video filename.

    Mirrors BOLT's `utils/mvbench_utils.py::_lmmseval_key`:
        os.path.basename(path).split('.')[0]
    i.e., truncate at the FIRST dot rather than `os.path.splitext` (which
    strips only the last extension). This matters for MVBench subtasks
    whose filenames embed timestamps —
        "ZS9XR_1.5_17.1.mp4".split('.')[0] == "ZS9XR_1"
    matches the JSON's video_id key, while os.path.splitext would yield
    "ZS9XR_1.5_17.1" which is not a JSON key.

    Backwards-compatible for the other benchmarks:
        "86CxyhFV9MI.mp4".split('.')[0] == "86CxyhFV9MI"          # LVB
        "0074f737-...-77c3a8127391.mp4".split('.')[0] == same     # EgoSchema
    """
    if not path or not isinstance(path, str):
        return ""
    return os.path.basename(path).split(".")[0]


# ----------------------------------------------------------------------
# ULR question-stem extraction helper
# ----------------------------------------------------------------------
_MCQ_CHOICE_RE = re.compile(
    r"\n\s*(?:[A-Da-d][\.\)]|\([A-Da-d]\)|[1-4][\.\)])\s+",
)
_ANSWER_INSTRUCTION_RE = re.compile(
    r"\b(?:answer with the (?:option'?s? )?letter|please answer "
    r"(?:with )?the correct option|choose (?:one|the correct)|"
    r"select the (?:correct|right) (?:answer|option))\b[^\n]*",
    re.IGNORECASE,
)


def _strip_to_question_stem(text: str) -> str:
    """Strip MCQ choices and trailing answer instruction.
    Keep only the question stem (matches offline parquet `question` field).
    Fallback: if no choice marker found, return trimmed full text.
    """
    if not text:
        return ""
    # Cut at first MCQ choice marker
    m = _MCQ_CHOICE_RE.search(text)
    if m:
        text = text[: m.start()].rstrip()
    # Drop trailing "Answer with ..." instruction if still present
    text = _ANSWER_INSTRUCTION_RE.sub("", text).strip()
    # Also drop any trailing "Question:" / "Q:" prefix that some tasks add
    text = re.sub(r"^\s*(?:Question|Q)\s*[:\-]\s*", "", text, flags=re.IGNORECASE)
    return text.strip()

from tqdm import tqdm

from lmms_eval.api.registry import register_model
from lmms_eval.imports import optional_import

VideoReader, _ = optional_import("decord", "VideoReader")
cpu, _ = optional_import("decord", "cpu")

from dotenv import load_dotenv
from loguru import logger as eval_logger

from lmms_eval.models.model_utils.gen_metrics import log_metrics
from lmms_eval.models.simple.openai_compatible import (
    OpenAICompatible as OpenAICompatibleSimple,
)
from lmms_eval.protocol import ChatMessages

load_dotenv(verbose=True)


@register_model("openai_compatible_chat")
class OpenAICompatible(OpenAICompatibleSimple):
    is_simple = False

    def __init__(self, *args, media_io_kwargs=None, **kwargs):
        # Strip keys that may be passed as JSON strings via model_args
        # so they don't reach OpenAICompatibleSimple.__init__ which only
        # accepts a subset.
        kwargs.pop("limit_mm_per_prompt", None)
        super().__init__(*args, **kwargs)

        # media_io_kwargs can come in as dict or JSON string (from CLI args)
        if isinstance(media_io_kwargs, str):
            import json as _json
            try:
                media_io_kwargs = _json.loads(media_io_kwargs)
            except Exception:
                media_io_kwargs = None
        self.media_io_kwargs = media_io_kwargs or None

    def _build_extra_body(self, per_request_video_kwargs=None):
        """Build extra_body for vLLM per-request overrides.

        Uses ChatMessages.build_openai_extra_body() which handles the
        whitelist-based kwarg filtering. If ``per_request_video_kwargs`` is
        given, it overlays on top of the static ``self.media_io_kwargs``
        (used e.g. for injecting the ULR question text per request).
        """
        base = (self.media_io_kwargs or {}).get("video") or {}
        video_kwargs = dict(base)
        if per_request_video_kwargs:
            video_kwargs.update(per_request_video_kwargs)
        if not video_kwargs:
            return None
        try:
            extra = ChatMessages.build_openai_extra_body(
                video_kwargs=video_kwargs,
                include_default_limit=True,
            )
        except Exception:
            extra = None
        return extra or None

    # ---- helpers ----
    @staticmethod
    def _strip_question_stem(text: str) -> str:
        """Strip MCQ choices and final instruction.  Kept for testing/reuse."""
        return _strip_to_question_stem(text)

    @staticmethod
    def _extract_question_text(chat_messages) -> str:
        """Pull the last user text turn, stripped to question stem only.

        Used by ULR (v14+) to inject the question into per-request
        ``ulr_question`` so the vLLM-side selector can route r_ipb.

        IMPORTANT: multiple-choice prompts contain "A. ... B. ... C. ... D. ..."
        choices after the question stem.  Those choices add noise to axis
        similarity scoring (e.g., the word "subtitle" or digits in answer text
        falsely activate OCR / count axes).  Offline ULR eval uses only the
        question stem (from parquet `question` field), so we must do the same
        here: strip everything from the first "A." / "a)" / "1." marker and
        drop the trailing "Answer with the option's letter ..." instruction.
        """
        try:
            for message in reversed(chat_messages.messages):
                if getattr(message, "role", None) != "user":
                    continue
                texts = [
                    c.text for c in message.content
                    if getattr(c, "type", None) == "text" and getattr(c, "text", None)
                ]
                if not texts:
                    continue
                full = " ".join(texts).strip()
                return _strip_to_question_stem(full)
        except Exception:
            pass
        return ""

    def generate_until(self, requests) -> List[str]:
        res = []

        batch_size = getattr(self, "batch_size_per_gpu", 1)
        batched_requests = [requests[i : i + batch_size] for i in range(0, len(requests), batch_size)]
        pbar = tqdm(
            total=len(batched_requests),
            disable=(self.rank != 0),
            desc="Model Responding",
        )

        e2e_latency = 0
        total_tokens = 0

        for batch_requests in batched_requests:
            batch_payloads = []
            batch_doc_uuids = []
            batch_responses = []

            for req in batch_requests:
                ctx, doc_to_messages, gen_kwargs, doc_id, task, split = req.args
                doc_uuid = f"{task}___{split}___{doc_id}"
                batch_doc_uuids.append(doc_uuid)

                if self.continual_mode is True and self.cache_mode == "resume":
                    if doc_uuid in self.response_cache:
                        response_text = self.response_cache[doc_uuid]
                        if response_text:
                            batch_responses.append(response_text)
                            continue

                chat_messages_raw = doc_to_messages(self.task_dict[task][split][doc_id])
                chat_messages: ChatMessages = ChatMessages(**{"messages": chat_messages_raw})

                payload = {"messages": chat_messages.to_openai_messages()}
                payload["model"] = self.model_version

                if "max_new_tokens" not in gen_kwargs:
                    gen_kwargs["max_new_tokens"] = 1024
                if gen_kwargs["max_new_tokens"] > 4096:
                    gen_kwargs["max_new_tokens"] = 4096
                if "temperature" not in gen_kwargs:
                    gen_kwargs["temperature"] = 0
                if "top_p" not in gen_kwargs:
                    gen_kwargs["top_p"] = None
                if "num_beams" not in gen_kwargs:
                    gen_kwargs["num_beams"] = 1

                payload["max_tokens"] = gen_kwargs["max_new_tokens"]
                payload["temperature"] = gen_kwargs["temperature"]

                if "o1" in self.model_version or "o3" in self.model_version or "o4" in self.model_version or "gpt-5" in self.model_version:
                    del payload["temperature"]
                    payload.pop("max_tokens")
                    # payload["reasoning_effort"] = "medium"
                    payload["response_format"] = {"type": "text"}
                    payload["max_completion_tokens"] = 5000

                per_req_video = {}
                video_kwargs_static = (self.media_io_kwargs or {}).get("video") or {}
                refine_mode = str(video_kwargs_static.get("ipb_refine_mode", "")).lower()
                video_backend = str(video_kwargs_static.get("video_backend", "")).lower()
                if refine_mode in ("v14", "v14_1", "v14_2", "v14_3", "v14_4",
                                    "v15_1", "v15_2", "v15_3", "v16",
                                    "v20_on", "v22_on", "v23_on", "v26_on",
                                    "v31", "v35_on",
                                    "v36_1", "v36_2_on", "v36_3"):
                    q_text = self._extract_question_text(chat_messages)
                    if q_text:
                        per_req_video["ulr_question"] = q_text

                # keyframe_json backend: inject (video_id, question_id) from
                # the doc so the vLLM-side backend can look up pre-computed
                # keyframes. Field names vary per task — try the union of
                # observed conventions:
                #   video_id  : video_id / videoID (videomme) / video_idx
                #               (egoschema) / video / video_path basename
                #   qid       : id (lvb) / question_id (videomme) / q_uid
                #               (egoschema) / qid
                if video_backend == "keyframe_json":
                    try:
                        doc = self.task_dict[task][split][doc_id]
                    except Exception:
                        doc = {}
                    if isinstance(doc, dict):
                        vid = (
                            doc.get("video_id")
                            or doc.get("videoID")
                            or doc.get("video_idx")
                            or _video_id_from_doc_path(doc.get("video_path"))
                            or _video_id_from_doc_path(doc.get("video"))
                        )
                        qid = (
                            doc.get("id")
                            or doc.get("question_id")
                            or doc.get("q_uid")
                            or doc.get("qid")
                        )
                        # MVBench rows have no unique id field at all
                        # (cols: video/question/candidates/answer). The JSON
                        # qid is `<subtask>_<row_idx>`; subtask comes from the
                        # task name (mvbench_<subtask>), row_idx == doc_id.
                        if not qid and isinstance(task, str) and task.startswith("mvbench_"):
                            qid = f"{task[len('mvbench_'):]}_{doc_id}"
                        if vid:
                            per_req_video["video_id"] = str(vid)
                        if qid is not None:
                            per_req_video["keyframe_question_id"] = str(qid)
                    # Always include question text as a fuzzy-match fallback.
                    if "ulr_question" not in per_req_video:
                        q_text = self._extract_question_text(chat_messages)
                        if q_text:
                            per_req_video["ulr_question"] = q_text

                extra_body = self._build_extra_body(per_request_video_kwargs=per_req_video or None)
                if extra_body:
                    payload["extra_body"] = extra_body

                batch_payloads.append(payload)
                batch_responses.append(None)

            def process_single_request(payload, i):
                if batch_responses[i] is not None:
                    return batch_responses[i], i, 0, 0

                for attempt in range(self.max_retries):
                    try:
                        start_time = time.time()
                        response = self.client.chat.completions.create(**payload)
                        end_time = time.time()

                        response_text = response.choices[0].message.content
                        latency = end_time - start_time

                        tokens = 0
                        if hasattr(response, "usage"):
                            tokens = response.usage.completion_tokens
                        else:
                            tokens = len(response_text.split())

                        return response_text, i, latency, tokens

                    except Exception as e:
                        error_msg = str(e)
                        eval_logger.info(f"Attempt {attempt + 1}/{self.max_retries} failed with error: {error_msg}")

                        if attempt == self.max_retries - 1:
                            eval_logger.error(f"All {self.max_retries} attempts failed. Last error: {error_msg}")
                            return "", i, 0, 0
                        else:
                            time.sleep(self.timeout)

                return "", i, 0, 0

            tasks_to_run = [(payload, i) for i, payload in enumerate(batch_payloads) if batch_responses[i] is None]

            if tasks_to_run:
                max_workers = min(len(tasks_to_run), 32)
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    future_to_index = {executor.submit(process_single_request, payload, i): i for payload, i in tasks_to_run}

                    for future in as_completed(future_to_index):
                        response_text, i, latency, tokens = future.result()
                        batch_responses[i] = response_text
                        e2e_latency += latency
                        total_tokens += tokens

            if self.continual_mode is True:
                for doc_uuid, response_text in zip(batch_doc_uuids, batch_responses):
                    if response_text is not None:
                        self.response_cache[doc_uuid] = response_text
                with open(self.response_persistent_file, "w") as f:
                    json.dump(self.response_cache, f)

            res.extend([r for r in batch_responses if r is not None])
            pbar.update(1)

        # Calculate average speed
        avg_speed = total_tokens / e2e_latency if e2e_latency > 0 else 0
        # Log metrics
        metric_dict = {
            "total_tokens": total_tokens,
            "e2e_latency": e2e_latency,
            "avg_speed": avg_speed,
        }
        log_metrics(**metric_dict)

        pbar.close()
        return res
