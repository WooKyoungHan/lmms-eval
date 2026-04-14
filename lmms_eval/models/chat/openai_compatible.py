import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

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

    @staticmethod
    def _extract_question_text(chat_messages) -> str:
        """Pull the last user text turn from a ChatMessages instance.

        Used by v14 ULR to inject the question into per-request
        ``ulr_question`` so the vLLM-side selector can route r_ipb.
        """
        try:
            for message in reversed(chat_messages.messages):
                if getattr(message, "role", None) != "user":
                    continue
                texts = [
                    c.text for c in message.content
                    if getattr(c, "type", None) == "text" and getattr(c, "text", None)
                ]
                if texts:
                    return " ".join(texts).strip()
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
                refine_mode = str(
                    ((self.media_io_kwargs or {}).get("video") or {}).get(
                        "ipb_refine_mode", ""
                    )
                ).lower()
                if refine_mode in ("v14", "v14_1", "v14_2", "v14_3", "v14_4",
                                    "v15_1", "v15_2", "v15_3", "v16",
                                    "v20_on", "v22_on", "v23_on", "v26_on"):
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
