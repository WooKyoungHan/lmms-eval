import re
from typing import List, Optional, Tuple, Union

import decord
import numpy as np
import torch
from accelerate import Accelerator, DistributedType
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    Qwen3VLForConditionalGeneration,
    Qwen3VLMoeForConditionalGeneration,
)

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.imports import optional_import
from lmms_eval.models.model_utils.reasoning_model_utils import (
    parse_reasoning_model_answer,
)

import os

from lmms_eval.frame_selectors.ipb_selector import (
    IPBSelectorConfig,
    select_frame_indices_ipb_propfair_gop,
)
from lmms_eval.frame_selectors.patch_selector import (
    PatchSelectorConfig,
    select_patch_positions_for_frames,
)


process_vision_info, _has_qwen_vl = optional_import("qwen_vl_utils", "process_vision_info")
if not _has_qwen_vl:
    eval_logger.warning("Failed to import qwen_vl_utils; Please install it via `pip install qwen-vl-utils`")


def _append_frame_stats(video_path: str, frame_indices: list[int]) -> None:
    frame_stats_path = os.environ.get("LMMS_FRAME_STATS_PATH", "").strip()
    if not frame_stats_path:
        return
    try:
        os.makedirs(os.path.dirname(frame_stats_path), exist_ok=True)
        import json
        with open(frame_stats_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "video": video_path,
                "num_selected": int(len(frame_indices)),
                "frame_indices": list(map(int, frame_indices)),
            }, ensure_ascii=False) + "\n")
    except Exception as e:
        eval_logger.warning(f"Failed to write frame stats to {frame_stats_path}: {e}")

def _normalize_mcqa_prompt(context: str) -> str:
    """
    Wrap MCQA prompts with the legacy VideoMME instruction format and
    remove explicit Question:/Options: section headers.
    """
    body = context.strip()
    if body.startswith("Question: "):
        body = body[len("Question: "):]
    body = body.replace("\nOptions:", "", 1)
    body = body.replace("Question:", "")
    body = body.replace("Options:", "")
    body = body.replace("\nAnswer with the option letter only.", "")
    body = body.replace("\nAnswer with the option letter only.", "")
    body = body.strip()
    return ("Select the best answer to the following multiple-choice question based on the video and the subtitles. Respond with only the letter (A, B, C, or D) of the correct option.\n"+ body + "\n\nAnswer with the option's letter from the given choices directly."
    )

def _env_flag(name: str, default: bool = False) -> bool:
    v = os.environ.get(name, None)
    if v is None:
        return default
    return str(v).strip().lower() in {"1", "true", "yes", "y", "on"}

def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except Exception:
        return default

def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except Exception:
        return default

def _build_ipb_video_payload(visual: str, max_pixels: int, min_pixels: int) -> dict:
    payload = {
        "type": "video",
        "video": visual,
        "max_pixels": max_pixels,
        "min_pixels": min_pixels,
    }
    algo = os.environ.get("LMMS_USE_ALGO", "").strip().lower()
    if not algo:
        return payload

    budget_ratio = _env_float("LMMS_BUDGET_RATIO", 1.0)
    utility = os.environ.get("LMMS_IPB_UTILITY", "v3")
    hybrid_uniform_ratio = _env_float("LMMS_HYBRID_UNIFORM_RATIO", 0.0)
    hybrid_min_dist_sec = _env_float("LMMS_HYBRID_MIN_DIST_SEC", 0.0)
    hybrid_max_refill_rounds = _env_int("LMMS_HYBRID_MAX_REFILL_ROUNDS", 0)

    cfg = IPBSelectorConfig(
        fps=_env_float("LMMS_VIDEO_FPS", 2.0),
        budget_ratio=budget_ratio,
        utility=utility,
        hybrid_uniform_ratio=hybrid_uniform_ratio,
        hybrid_min_dist_sec=hybrid_min_dist_sec,
        hybrid_max_refill_rounds=hybrid_max_refill_rounds,
    )
    frame_indices = select_frame_indices_ipb_propfair_gop(visual, cfg)
    payload["frame_indices"] = list(map(int, frame_indices))
    _append_frame_stats(visual, payload["frame_indices"])

    if algo == "ipb_v4":
        patch_cfg = PatchSelectorConfig(
            patch_size=_env_int("LMMS_CODEC_PATCH_SIZE", 16),
            square_size=_env_int("LMMS_CODEC_SQUARE_SIZE", 576),
            keep_ratio=_env_float("LMMS_CODEC_PATCH_KEEP_RATIO", 0.125),
            num_patches_per_frame=(
                None if os.environ.get("LMMS_CODEC_NUM_PATCHES_PER_FRAME") in (None, "", "none", "None")
                else _env_int("LMMS_CODEC_NUM_PATCHES_PER_FRAME", 0)
            ),
            iframe_full=_env_flag("LMMS_CODEC_IFRAME_FULL", True),
        )
        patch_out = select_patch_positions_for_frames(
            video_path=visual,
            selected_frame_indices=payload["frame_indices"],
            cfg=patch_cfg,
        )
        payload["codec_token_prune"] = True
        payload["codec_keep_thw"] = patch_out["patch_positions"]
        payload["codec_patch_size"] = patch_cfg.patch_size
        payload["codec_iframe_full"] = patch_cfg.iframe_full
        payload["codec_frame_types"] = patch_out.get("frame_types", None)
        payload["codec_coord_space"] = "square_grid"
    return payload


@register_model("qwen3_vl")
class Qwen3_VL(lmms):
    """
    Qwen3_VL Model
    "https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct"
    """

    def __init__(
        self,
        pretrained: str = "Qwen/Qwen3-VL-4B-Instruct",
        device: Optional[str] = "cuda",
        device_map: Optional[str] = "auto",
        batch_size: Optional[Union[int, str]] = 1,
        use_cache=True,
        attn_implementation: Optional[str] = None,
        min_pixels: int = 256 * 28 * 28,
        max_pixels: int = 1605632,
        max_num_frames: int = 32,
        use_custom_video_loader: Optional[bool] = False,
        fps: Optional[float] = None,  # Only applicable if use_custom_video_loader is True
        max_image_size: Optional[int] = None,  # Only applicable if use_custom_video_loader is True
        system_prompt: Optional[str] = "",
        interleave_visuals: Optional[bool] = False,
        reasoning_prompt: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        # Do not use kwargs for now
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        # Validate attention implementation
        valid_attn_implementations = [None, "flash_attention_2", "sdpa", "eager"]
        if attn_implementation not in valid_attn_implementations:
            raise ValueError(f"attn_implementation must be one of {valid_attn_implementations}, got {attn_implementation}")

        self.use_custom_video_loader = use_custom_video_loader
        self.fps = fps
        # if self.fps and not self.use_custom_video_loader:
        #     raise ValueError("FPS is only applicable if use_custom_video_loader is True")
        self.max_image_size = max_image_size
        if self.max_image_size and not self.use_custom_video_loader:
            raise ValueError("max_image_size is only applicable if use_custom_video_loader is True")

        accelerator = Accelerator()
        self.accelerator = accelerator
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        else:
            self._device = torch.device(device)
            self.device_map = device_map if device_map else device

        # Prepare model loading arguments
        model_kwargs = {
            "dtype": "bfloat16",
            "device_map": self.device_map,
        }

        # Add attention implementation if specified
        if attn_implementation is not None:
            model_kwargs["attn_implementation"] = attn_implementation

        # check whether its an MoE model
        match = re.search(r"A\d+B", pretrained)
        model_fn = Qwen3VLMoeForConditionalGeneration if match else Qwen3VLForConditionalGeneration
        self._model = model_fn.from_pretrained(pretrained, **model_kwargs).eval()
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels
        self.max_num_frames = max_num_frames

        if reasoning_prompt:
            self.reasoning_prompt = reasoning_prompt.replace("\\n", "\n")
        else:
            self.reasoning_prompt = None
        self.processor = AutoProcessor.from_pretrained(pretrained, max_pixels=max_pixels, min_pixels=min_pixels)
        self._tokenizer = AutoTokenizer.from_pretrained(pretrained)
        self.system_prompt = system_prompt
        self.interleave_visuals = interleave_visuals

        self._config = self.model.config
        self._max_length = kwargs.get("max_length", 2048)
        self.batch_size_per_gpu = int(batch_size)
        self.use_cache = use_cache

        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [
                DistributedType.FSDP,
                DistributedType.MULTI_GPU,
            ], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            if accelerator.distributed_type == DistributedType.FSDP:
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(self.model, evaluation_mode=True)
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self._rank = 0
            self._world_size = 1

    @property
    def config(self):
        # return the associated transformers.AutoConfig for the given pretrained model.
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        # returns the model, unwrapping it if using Accelerate
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        else:
            return self._model

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_length

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("Loglikelihood is not implemented for Qwen2.5_VL")

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []

        def _collate(x):
            # the negative sign on len(toks) sorts descending - this has a few advantages:
            # - time estimates will always be over not underestimates, which is more useful for planning
            # - to know the size of a batch when going through the list, you know the first one is always the batch
            #   padded context length. this is useful to simplify the batching logic and more importantly to make
            #   automatic adaptive batches much much easier to implement
            # - any OOMs will happen right away rather than near the end
            toks = self.tokenizer.encode(x[0])
            return -len(toks), x[0]

        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")
        # we group requests by their generation_kwargs,
        # so that we don't try to execute e.g. greedy sampling and temp=0.8 sampling
        # in the same batch.
        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        for chunk in chunks:
            contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
            task = task[0]
            split = split[0]
            visual_list = [doc_to_visual[0](self.task_dict[task][split][ids]) for ids in doc_id]
            gen_kwargs = all_gen_kwargs[0]

            # Set default until or update values from gen_kwargs if present
            until = gen_kwargs.get("until", [self.tokenizer.decode(self.eot_token_id)])

            if isinstance(until, str):
                until = [until]
            elif not isinstance(until, list):
                raise ValueError(f"Expected `gen_kwargs['until']` to be of type Union[str, list], but got {type(until)}")

            # Avoid using '\n\n' as a stopper for Qwen2.5VL to prevent truncation, which can lead to incorrect results
            until = [item for item in until if item != "\n\n"]

            if isinstance(contexts, tuple):
                contexts = list(contexts)

            for i in range(len(contexts)):
                if "<image>" in contexts[i]:
                    contexts[i] = contexts[i].replace("<image>", "")

            batched_messages = []
            for i, context in enumerate(contexts):
                if "<image>" in context:
                    context = context.replace("<image>", "")
                # print(context)
                context = _normalize_mcqa_prompt(context)
                # print(context)
                contexts[i] = context
                message = []
                if self.system_prompt:
                    message.append({"role": "system", "content": self.system_prompt})
                if self.reasoning_prompt:
                    context = context.strip() + self.reasoning_prompt
                    contexts[i] = context

                processed_visuals = []
                if visual_list[i] is not None:
                    for visual in visual_list[i]:
                        if isinstance(visual, str) and visual.endswith((".mp4", ".avi", ".mov")):  # Video file
                            vr = decord.VideoReader(visual)
                            first_frame = vr[0].asnumpy()
                            height, width = first_frame.shape[:2]
                            # max_pixels = height * width
                            processed_visuals.append(
                                _build_ipb_video_payload(
                                    visual=visual,
                                    max_pixels=self.max_pixels,
                                    min_pixels=self.min_pixels,
                                )
                            )
                        elif isinstance(visual, Image.Image):  # Handle both single and multiple images
                            processed_visuals.append(
                                {
                                    "type": "image",
                                    "image": visual,
                                    "max_pixels": self.max_pixels,
                                    "min_pixels": self.min_pixels,
                                }
                            )

                if self.interleave_visuals is False:
                    message.append(
                        {
                            "role": "user",
                            "content": processed_visuals + [{"type": "text", "text": context}],
                        }
                    )
                else:  # currently support find <image x> in the context
                    image_placeholders = re.findall(r"<image \d+>", context)
                    content_parts = []
                    text_parts = re.split(r"<image \d+>", context)
                    if text_parts[0]:
                        content_parts.append({"type": "text", "text": text_parts[0]})

                    for i, placeholder in enumerate(image_placeholders):
                        img_idx = int(re.search(r"<image (\d+)>", placeholder).group(1)) - 1
                        image_idx = min(img_idx, len(processed_visuals) - 1) if processed_visuals else 0
                        if processed_visuals and image_idx < len(processed_visuals):
                            content_parts.append(processed_visuals[image_idx])
                        if i + 1 < len(text_parts) and text_parts[i + 1]:
                            content_parts.append({"type": "text", "text": text_parts[i + 1]})

                    message.append(
                        {
                            "role": "user",
                            "content": content_parts,
                        }
                    )

                batched_messages.append(message)
            texts = self.processor.apply_chat_template(batched_messages, tokenize=False, add_generation_prompt=True)
            # TODO: refactor code to allow return_video_kwargs and return_video_metadata
            image_inputs, video_inputs = process_vision_info(
                batched_messages,
                return_video_kwargs=False,
                image_patch_size=16,
                return_video_metadata=False,
            )

            codec_keep_thw = []
            codec_token_prune = False
            for message in batched_messages:
                for content in message:
                    if content["role"] != "user":
                        continue
                    for part in content["content"]:
                        if isinstance(part, dict) and part.get("type") == "video":
                            if part.get("codec_token_prune", False):
                                codec_token_prune = True
                            codec_keep_thw.append(part.get("codec_keep_thw", None))

            if video_inputs is not None:
                eval_logger.info(f"[qwen3_vl] keeping all selected decoded frames: {tuple(video_inputs[0].shape)}")
            if self.batch_size > 1:
                inputs = self.processor(
                    text=texts,
                    images=image_inputs,
                    videos=video_inputs,
                    do_resize=False,
                    do_sample_frames=False,
                    padding=True,
                    padding_side="left",
                    return_tensors="pt",
                    codec_token_prune=codec_token_prune,
                    codec_keep_thw=codec_keep_thw if codec_keep_thw else None,
                )
            else:
                inputs = self.processor(
                    text=texts,
                    images=image_inputs,
                    videos=video_inputs,
                    do_resize=False,
                    do_sample_frames=False,
                    return_tensors="pt",
                    codec_token_prune=codec_token_prune,
                    codec_keep_thw=codec_keep_thw if codec_keep_thw else None,
                )
            if self.device_map == "auto":
                inputs = inputs.to("cuda")
            else:
                inputs = inputs.to(self.device)
            print("[DBG][MODEL_INPUT] pixel_values_videos.shape =", tuple(inputs["pixel_values_videos"].shape))
            print("[DBG][MODEL_INPUT] video_grid_thw =", inputs.get("video_grid_thw", None))
            print("[DBG][MODEL_INPUT] image_processor =", self.processor.image_processor)            # Set default generation kwargs
            print("[DBG][BEFORE-PROCESSOR] video_inputs[0].shape =", tuple(video_inputs[0].shape))
            print("[DBG][MODEL_INPUT] video_grid_thw =", inputs["video_grid_thw"])  
            default_gen_kwargs = {
                "max_new_tokens": 128,
                "temperature": 0.0,  # Set to 0 for greedy default
                "top_p": None,
                "num_beams": 1,
            }
            # Update with provided kwargs
            current_gen_kwargs = {**default_gen_kwargs, **gen_kwargs}
            pad_token_id = self.tokenizer.pad_token_id

            if current_gen_kwargs["temperature"] > 0:
                current_gen_kwargs["do_sample"] = True
            else:
                current_gen_kwargs["do_sample"] = False
                current_gen_kwargs["temperature"] = None
                current_gen_kwargs["top_p"] = None

            cont = self.model.generate(
                **inputs,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=pad_token_id,
                do_sample=current_gen_kwargs["do_sample"],
                temperature=current_gen_kwargs["temperature"],
                top_p=current_gen_kwargs["top_p"],
                num_beams=current_gen_kwargs["num_beams"],
                max_new_tokens=current_gen_kwargs["max_new_tokens"],
                use_cache=self.use_cache,
            )

            generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, cont)]
            answers = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            for i, ans in enumerate(answers):
                for term in until:
                    if len(term) > 0:
                        ans = ans.split(term)[0]
                answers[i] = ans

            for ans, context in zip(answers, contexts):
                clean_ans = parse_reasoning_model_answer(ans)
                res.append(clean_ans)
                self.cache_hook.add_partial("generate_until", (context, gen_kwargs), clean_ans)
                pbar.update(1)

                # eval_logger.debug(f"Question: {context}")
                # eval_logger.debug(f"Model Raw Response: {ans}")
                # eval_logger.debug(f"Model Clean Response: {clean_ans}")
            # reorder this group of results back to original unsorted form
        res = re_ords.get_original(res)

        pbar.close()
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation")
