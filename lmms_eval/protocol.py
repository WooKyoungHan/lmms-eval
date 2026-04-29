import base64
import os
import time
import json
import atexit
from io import BytesIO
from typing import Any, Dict, List, Literal, Union, Optional, Tuple

import numpy as np
from PIL import Image
from pydantic import BaseModel

from lmms_eval.imports import optional_import

from lmms_eval.frame_selectors.ipb_selector import (
    IPBSelectorConfig,
    select_frame_indices_ipb,
    select_frame_indices_ipb_propfair_gop,
)
from lmms_eval.frame_selectors.patch_selector import (
    PatchSelectorConfig,
    select_patch_positions_for_frames,
)

# Optional video processing dependencies
VideoReader, _has_decord = optional_import("decord", "VideoReader")
cpu, _ = optional_import("decord", "cpu")
fetch_video, _has_qwen_vl = optional_import("qwen_vl_utils", "fetch_video")

_FRAME_STATS = {"records": []}
_BASE_FPS = 2.0


def _resolve_budget_frames(video_kwargs: Dict[str, Any]) -> Optional[int]:
    """
    Priority:
      1) video_kwargs["budget_frames"]
      2) env LMMS_BUDGET_FRAMES
      3) None
    """
    v = (video_kwargs or {}).get("budget_frames", None)
    try:
        if v is not None:
            iv = int(v)
            if iv > 0:
                return iv
    except Exception:
        pass

    env = os.environ.get("LMMS_BUDGET_FRAMES", "").strip()
    try:
        if env:
            iv = int(env)
            if iv > 0:
                return iv
    except Exception:
        pass

    return None


def _validate_budget_controls(video_kwargs: Dict[str, Any]) -> None:
    br_from_kwargs = _coerce_float((video_kwargs or {}).get("budget_ratio", None))
    k_from_kwargs = (video_kwargs or {}).get("budget_frames", None)

    has_br_kwargs = br_from_kwargs is not None
    has_k_kwargs = False
    try:
        has_k_kwargs = (k_from_kwargs is not None) and (int(k_from_kwargs) > 0)
    except Exception:
        has_k_kwargs = False

    br_env = _coerce_float(os.environ.get("LMMS_BUDGET_RATIO", "").strip() or None)
    k_env_raw = os.environ.get("LMMS_BUDGET_FRAMES", "").strip()
    has_br_env = br_env is not None
    has_k_env = False
    try:
        has_k_env = bool(k_env_raw) and (int(k_env_raw) > 0)
    except Exception:
        has_k_env = False

    has_br = has_br_kwargs or has_br_env
    has_k = has_k_kwargs or has_k_env

    if has_br and has_k:
        raise ValueError("budget_ratio and budget_frames cannot be used together")

def _resolve_hybrid_uniform_ratio(video_kwargs: Dict[str, Any]) -> float:
    """
    Priority:
      1) video_kwargs["hybrid_uniform_ratio"]
      2) env LMMS_HYBRID_UNIFORM_RATIO
      3) default 0.0
    """
    v = (video_kwargs or {}).get("hybrid_uniform_ratio", None)
    try:
        if v is not None:
            v = float(v)
            return max(0.0, min(1.0, v))
    except Exception:
        pass

    env = os.environ.get("LMMS_HYBRID_UNIFORM_RATIO", "").strip()
    try:
        if env:
            env = float(env)
            return max(0.0, min(1.0, env))
    except Exception:
        pass

    return 0.0


def _resolve_hybrid_min_dist_sec(video_kwargs: Dict[str, Any]) -> Optional[float]:
    """
    Priority:
      1) video_kwargs["hybrid_min_dist_sec"]
      2) env LMMS_HYBRID_MIN_DIST_SEC
      3) None
    """
    v = _coerce_float((video_kwargs or {}).get("hybrid_min_dist_sec", None))
    if v is not None:
        return float(v)

    env = _coerce_float(os.environ.get("LMMS_HYBRID_MIN_DIST_SEC", "").strip() or None)
    if env is not None:
        return float(env)

    return None


def _resolve_hybrid_max_refill_rounds(video_kwargs: Dict[str, Any]) -> int:
    """
    Priority:
      1) video_kwargs["hybrid_max_refill_rounds"]
      2) env LMMS_HYBRID_MAX_REFILL_ROUNDS
      3) 8
    """
    v = (video_kwargs or {}).get("hybrid_max_refill_rounds", None)
    try:
        if v is not None:
            v = int(v)
            return max(1, v)
    except Exception:
        pass

    env = os.environ.get("LMMS_HYBRID_MAX_REFILL_ROUNDS", "").strip()
    try:
        if env:
            env = int(env)
            return max(1, env)
    except Exception:
        pass

    return 8

def _dump_frame_stats():
    path = os.environ.get("LMMS_FRAME_STATS_PATH", "").strip()
    if not path:
        return
    recs = _FRAME_STATS.get("records", [])
    if not recs:
        return

    summary = {
        "n": len(recs),
        "total_frames_used": sum(r["frames_used"] for r in recs),
        "mean_frames_used": sum(r["frames_used"] for r in recs) / len(recs),
        "max_frames_used": max(r["frames_used"] for r in recs),
        "min_frames_used": min(r["frames_used"] for r in recs),
        "algo_on_count": sum(1 for r in recs if r["algo"] in {"ipb", "ipb_v2"}),
        "algo_off_count": sum(1 for r in recs if r["algo"] == "default"),
    }

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump({"summary": summary, "records": recs}, f, indent=2)


atexit.register(_dump_frame_stats)


def _record_frames_used(video_path: str, algo: str, fps: float, frames_used: int):
    _FRAME_STATS["records"].append(
        {
            # "ts": time.time(),
            "video": str(video_path),
            "algo": algo,  # "ipb" or "default"
            "fps": float(fps),
            "frames_used": int(frames_used),
        }
    )


def _coerce_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        v = float(x)
        if v > 0:
            return v
    except Exception:
        pass
    return None


def _resolve_algo(video_kwargs: Dict[str, Any]) -> str:
    """
    Decide frame selection algorithm.
    Priority:
      1) video_kwargs["frame_selector"] if present
      2) env LMMS_USE_ALGO
      3) default
    """
    v = (video_kwargs or {}).get("frame_selector", None)
    if isinstance(v, str) and v.strip():
        v = v.strip().lower()
        if v in {"ipb", "ipb_v2", "ipb_v4"}:
            return v
        return "default"

    env = os.environ.get("LMMS_USE_ALGO", "").strip().lower()
    if env in {"ipb", "ipb_v2", "ipb_v4", "default"}:
        return env
    return "default"

def _resolve_budget_ratio(video_kwargs: Dict[str, Any]) -> float:
    """
    Priority:
      1) video_kwargs["budget_ratio"]
      2) env LMMS_BUDGET_RATIO
      3) default 1.0
    """
    v = _coerce_float((video_kwargs or {}).get("budget_ratio", None))
    if v is not None:
        return float(v)
    env = _coerce_float(os.environ.get("LMMS_BUDGET_RATIO", "").strip() or None)
    if env is not None:
        return float(env)
    return 1.0



def _resolve_patch_keep_ratio(video_kwargs: Dict[str, Any]) -> float:
    v = (video_kwargs or {}).get("codec_patch_keep_ratio", None)
    try:
        if v is not None:
            v = float(v)
            return max(0.0, min(1.0, v))
    except Exception:
        pass
    env = os.environ.get("LMMS_CODEC_PATCH_KEEP_RATIO", "").strip()
    try:
        if env:
            env = float(env)
            return max(0.0, min(1.0, env))
    except Exception:
        pass
    return 0.125


def _resolve_patch_num_per_frame(video_kwargs: Dict[str, Any]) -> Optional[int]:
    v = (video_kwargs or {}).get("codec_num_patches_per_frame", None)
    try:
        if v is not None:
            return max(1, int(v))
    except Exception:
        pass
    env = os.environ.get("LMMS_CODEC_NUM_PATCHES_PER_FRAME", "").strip()
    try:
        if env:
            return max(1, int(env))
    except Exception:
        pass
    return None


def _resolve_iframe_full(video_kwargs: Dict[str, Any]) -> bool:
    v = (video_kwargs or {}).get("codec_iframe_full", None)
    if isinstance(v, bool):
        return v
    if isinstance(v, str) and v.strip():
        return v.strip().lower() in {"1", "true", "yes", "y", "on"}
    env = os.environ.get("LMMS_CODEC_IFRAME_FULL", "").strip().lower()
    if env:
        return env in {"1", "true", "yes", "y", "on"}
    return True


def _resolve_patch_size(video_kwargs: Dict[str, Any]) -> int:
    v = (video_kwargs or {}).get("codec_patch_size", None)
    try:
        if v is not None:
            return max(1, int(v))
    except Exception:
        pass
    env = os.environ.get("LMMS_CODEC_PATCH_SIZE", "").strip()
    try:
        if env:
            return max(1, int(env))
    except Exception:
        pass
    return 16


def _resolve_square_size(video_kwargs: Dict[str, Any]) -> int:
    v = (video_kwargs or {}).get("codec_square_size", None)
    try:
        if v is not None:
            return max(16, int(v))
    except Exception:
        pass
    env = os.environ.get("LMMS_CODEC_SQUARE_SIZE", "").strip()
    try:
        if env:
            return max(16, int(env))
    except Exception:
        pass
    return 576


def _build_ipb_cfg(video_kwargs: Dict[str, Any], budget_ratio: float) -> IPBSelectorConfig:
    utility = _resolve_utility(video_kwargs)
    alpha, beta = _resolve_alpha_beta(video_kwargs)
    hybrid_uniform_ratio = _resolve_hybrid_uniform_ratio(video_kwargs)
    hybrid_min_dist_sec = _resolve_hybrid_min_dist_sec(video_kwargs)
    hybrid_max_refill_rounds = _resolve_hybrid_max_refill_rounds(video_kwargs)
    budget_frames = _resolve_budget_frames(video_kwargs)

    return IPBSelectorConfig(
        fps=float(_BASE_FPS),
        budget_ratio=budget_ratio,
        budget_frames=budget_frames,
        utility=utility,
        alpha=alpha,
        beta=beta,
        hybrid_uniform_ratio=hybrid_uniform_ratio,
        hybrid_min_dist_sec=hybrid_min_dist_sec,
        hybrid_max_refill_rounds=hybrid_max_refill_rounds,
    )

def _apply_frame_and_patch_selection_to_payload(
    video_url: str,
    payload: Dict[str, Any],
    video_kwargs: Dict[str, Any],
    algo: str,
    budget_ratio: float,
) -> None:
    _validate_budget_controls(video_kwargs)
    budget_frames = _resolve_budget_frames(video_kwargs)

    if algo == "default":
        if budget_frames is not None:
            payload["frame_budget"] = int(budget_frames)
        else:
            payload["fps"] = float(_BASE_FPS * budget_ratio)
        return

    cfg = _build_ipb_cfg(video_kwargs, budget_ratio)

    if algo == "ipb":
        frame_indices = select_frame_indices_ipb(video_url, cfg)
    else:
        frame_indices = select_frame_indices_ipb_propfair_gop(video_url, cfg)

    payload["frame_indices"] = frame_indices

    if algo == "ipb_v4":
        patch_cfg = PatchSelectorConfig(
            patch_size=_resolve_patch_size(video_kwargs),
            square_size=_resolve_square_size(video_kwargs),
            keep_ratio=_resolve_patch_keep_ratio(video_kwargs),
            num_patches_per_frame=_resolve_patch_num_per_frame(video_kwargs),
            iframe_full=_resolve_iframe_full(video_kwargs),
        )
        patch_out = select_patch_positions_for_frames(
            video_path=video_url,
            selected_frame_indices=frame_indices,
            cfg=patch_cfg,
        )
        payload["codec_patchify"] = True
        payload["codec_patch_positions"] = patch_out["patch_positions"]
        payload["codec_patch_keep_ratio"] = patch_cfg.keep_ratio
        payload["codec_num_patches_per_frame"] = patch_cfg.num_patches_per_frame
        payload["codec_iframe_full"] = patch_cfg.iframe_full
        payload["codec_patch_size"] = patch_cfg.patch_size
        payload["codec_square_size"] = patch_cfg.square_size


def _resolve_fps(video_kwargs: Dict[str, Any]) -> Optional[float]:
    """
    Decide sampling FPS.
    Priority:
      1) video_kwargs["fps"] if present
      2) env LMMS_VIDEO_FPS
      3) None (let downstream decide)
    """
    v = _coerce_float((video_kwargs or {}).get("fps", None))
    if v is not None:
        return v

    env = _coerce_float(os.environ.get("LMMS_VIDEO_FPS", "").strip() or None)
    if env is not None:
        return env
    return None


def _get_active_model_name(video_kwargs: Dict[str, Any]) -> str:
    video_kwargs = video_kwargs or {}
    for key in ("model", "model_name", "pretrained"):
        value = video_kwargs.get(key, None)
        if isinstance(value, str) and value.strip():
            return value.strip().lower()

    for env_key in ("MODEL", "MODEL_NAME", "HF_MODEL_ID", "VLLM_MODEL"):
        value = os.environ.get(env_key, "")
        if isinstance(value, str) and value.strip():
            return value.strip().lower()

    return ""


def _should_emit_video_url_for_model(video_kwargs: Dict[str, Any]) -> bool:
    model_name = _get_active_model_name(video_kwargs)
    if not model_name:
        return False
    return (
        "llava-onevision" in model_name
        or "llava_ov" in model_name
        or "llava-ov" in model_name
        or ("llava" in model_name and "onevision" in model_name)
    )


def _normalize_openai_video_url(video_url: Any) -> Any:
    if not isinstance(video_url, str):
        return video_url
    if video_url.startswith(("http://", "https://", "file://", "data:")):
        return video_url
    return f"file://{os.path.abspath(video_url)}"

def _resolve_utility(video_kwargs: Dict[str, Any]) -> str:
    """third_party/lmms-eval/lmms_eval/frame_selectors/__pycache__
    Decide IPB utility function for GOP allocator.
    Priority:
      1) video_kwargs["utility"] if present
      2) env LMMS_IPB_UTILITY
      3) "log"
    """
    v = (video_kwargs or {}).get("utility", None)
    if isinstance(v, str) and v.strip():
        v = v.strip().lower()
        if v in {"log", "alpha_fair", "exp","v3"}:
            return v

    env = os.environ.get("LMMS_IPB_UTILITY", "").strip().lower()
    if env in {"log", "alpha_fair", "exp","v3"}:
        return env
    return "log"


def _resolve_alpha_beta(video_kwargs: Dict[str, Any]) -> Tuple[float, float]:
    """
    Optional knobs for alpha_fair / exp utilities.
    Priority:
      1) video_kwargs["alpha"/"beta"]
      2) env LMMS_IPB_ALPHA / LMMS_IPB_BETA
      3) defaults (alpha=2.0, beta=0.5)
    """
    alpha = _coerce_float((video_kwargs or {}).get("alpha", None)) or _coerce_float(os.environ.get("LMMS_IPB_ALPHA", "").strip() or None) or 2.0
    beta = _coerce_float((video_kwargs or {}).get("beta", None)) or _coerce_float(os.environ.get("LMMS_IPB_BETA", "").strip() or None) or 0.5
    return float(alpha), float(beta)


class ChatTextContent(BaseModel):
    type: Literal["text"] = "text"
    text: str


class ChatImageContent(BaseModel):
    type: Literal["image"] = "image"
    url: Any


class ChatVideoContent(BaseModel):
    type: Literal["video"] = "video"
    url: Any


class ChatAudioContent(BaseModel):
    type: Literal["audio"] = "audio"
    url: Any


ChatContent = Union[ChatTextContent, ChatImageContent, ChatVideoContent, ChatAudioContent]


class ChatMessage(BaseModel):
    role: Literal["user", "system", "assistant"]
    content: List[ChatContent]


class ChatMessages(BaseModel):
    messages: List[ChatMessage]

    @staticmethod
    def build_openai_extra_body(
        video_kwargs: Dict[str, Any] | None = None,
        extra_body: Dict[str, Any] | None = None,
        include_default_limit: bool = True,
    ) -> Dict[str, Any]:
        """
        Build runtime extra_body for vLLM OpenAI-compatible requests.

        This is the correct place for request-level video backend / budget controls.
        The OpenAI message payload itself should only carry the video URL.
        """
        video_kwargs = dict(video_kwargs or {})
        out = dict(extra_body or {})

        runtime_video_keys = {
            "video_backend",
            "num_frames",
            "budget_frames",
            "budget_fps",
            "fps",
            "max_duration",
            "frame_indices",
            "frames_indices",
            "frame_recovery",
            "return_selector_debug",
            # IPB selector controls (per-request override)
            "ipb_refine_mode",
            "r_ipb",
            "c_min_dist",
            # oracle_v2 / oracle_v3 (anchor-replacement oracle builders)
            "discount_type",
            "lam",
            "gamma",
            "min_gap_sec",
            "eta_max_gap",
            # v13 density parameters
            "v13_alpha",
            "v13_beta",
            "v13_sigma_sec",
            "v13_sigma_adaptive",
            # v37 density parameters (promotion of v13)
            "v37_alpha",
            "v37_beta",
            "v37_sigma_sec",
            "v37_sigma_adaptive",
            # v38: v37 + GOP duration-drop pre-filtering
            "v38_alpha",
            "v38_beta",
            "v38_sigma_sec",
            "v38_sigma_adaptive",
            "v38_min_gop_sec",
            # v14 ULR: per-request question text for text→r routing
            "ulr_question",
            # v5 focus-context parameters
            "v5_n_seg",
            "v5_n_focus",
            "v5_seg_width_mult",
            "v5_n_best",
            "v5_n_worst",
            # v6 residual-surprise parameters
            "v6_n_seg",
            "v6_n_worst",
            "v6_n_focus",
            "v6_alpha",
            "v6_seg_width_mult",
            # v40 parameters
            "mcv_threshold",
            # v72 parameters
            "v72_quantile",
            "v72_min_seg_bins",
            "v72_gap_merge_bins",
            "v72_kappa",
            "v72_density_mult",
            "v72_gamma",
            "v72_max_peaks",
            "v72_kappa_amp",
            "v72_kappa_tau",
            "v72_kappa_base",
            "v72_bin_ref_sec",
            "v72_bin_c",
            "v82_max_depth",
            "v82_t1",
            "v82_lam",
            "v82_n_bins",
            "v91_sub_mult",
            "v91_ref_K_coarse",
            "v91_ref_K_fine",
            "v91_q_fine_offset",
            # generic per-request overrides
            "ot_lam",
            "coverage_c",
            # codec-driven per-patch mask (opt-in; default OFF)
            "codec_mask_mode",
            "codec_mask_keep_ratio",
            "codec_mask_iframe_full",
            "codec_mask_score",
            "codec_mask_patch_size",
            "codec_mask_budget_path",
            "codec_mask_drop_stage",
            # keyframe_json backend (pre-computed per-(video,question) keyframes)
            "video_id",
            "keyframe_question_id",
        }

        runtime_video_kwargs = {
            k: v for k, v in video_kwargs.items()
            if k in runtime_video_keys and v is not None
        }

        if runtime_video_kwargs:
            out["media_io_kwargs"] = {"video": runtime_video_kwargs}

        if include_default_limit and "limit_mm_per_prompt" not in out:
            out["limit_mm_per_prompt"] = {"video": -1}

        return out

    def extract_media(self):
        images = []
        videos = []
        audios = []

        for message in self.messages:
            for content in message.content:
                if content.type == "image":
                    images.append(content.url)
                elif content.type == "video":
                    videos.append(content.url)
                elif content.type == "audio":
                    audios.append(content.url)

        return images, videos, audios

    def to_hf_messages(self, video_kwargs: Dict[str, Any] = None):
        if video_kwargs is None:
            video_kwargs = {}
        _num_frames = video_kwargs.get("nframes", 32)  # noqa: F841
        hf_messages = []
        for message in self.messages:
            hf_message = {"role": message.role, "content": []}
            for content in message.content:
                if content.type == "text":
                    hf_message["content"].append({"type": "text", "text": content.text})
                elif content.type == "image":
                    hf_message["content"].append({"type": "image", "image": content.url})
                elif content.type == "video":
                    hf_message["content"].append({"type": "video", "video": content.url, **video_kwargs})
                elif content.type == "audio":
                    hf_message["content"].append({"type": "audio", "audio": content.url})
            hf_messages.append(hf_message)
        return hf_messages

    def to_openai_messages(self, video_kwargs: Dict[str, Any] = None):
        # NOTE:
        # video_kwargs is intentionally NOT embedded into the OpenAI message body.
        # Request-level video controls must go through build_openai_extra_body(...).
        if video_kwargs is None:
            video_kwargs = {}

        openai_messages = []
        for message in self.messages:
            openai_message = {"role": message.role, "content": []}
            for content in message.content:
                if content.type == "text":
                    openai_message["content"].append(
                        {"type": "text", "text": content.text}
                    )
                elif content.type == "image":
                    openai_message["content"].append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{self.encode_image(content.url)}"
                            },
                        }
                    )
                elif content.type == "video":
                    openai_message["content"].append(
                        {
                            "type": "video_url",
                            "video_url": {
                                "url": _normalize_openai_video_url(content.url),
                            },
                        }
                    )
                elif content.type == "audio":
                    openai_message["content"].append(
                        {"type": "audio_url", "audio_url": {"url": content.url}}
                    )

            openai_messages.append(openai_message)

        return openai_messages

    def to_qwen3_vl_openai_messages(self, video_kwargs: Dict[str, Any] = None):
        # NOTE:
        # video_kwargs is intentionally NOT embedded into the OpenAI message body.
        # Request-level video controls must go through build_openai_extra_body(...).
        if video_kwargs is None:
            video_kwargs = {}

        openai_messages = []
        for message in self.messages:
            openai_message = {"role": message.role, "content": []}
            for content in message.content:
                if content.type == "text":
                    openai_message["content"].append(
                        {"type": "text", "text": content.text}
                    )
                elif content.type == "image":
                    openai_message["content"].append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{self.encode_image(content.url)}"
                            },
                        }
                    )
                elif content.type == "video":
                    openai_message["content"].append(
                        {
                            "type": "video_url",
                            "video_url": {
                                "url": _normalize_openai_video_url(content.url),
                            },
                        }
                    )
                elif content.type == "audio":
                    openai_message["content"].append(
                        {"type": "audio_url", "audio_url": {"url": content.url}}
                    )

            openai_messages.append(openai_message)

        return openai_messages

    def _calculate_timestamps(self, video_metadata: Dict[str, Any]):
        indices = video_metadata["frames_indices"]
        if not isinstance(indices, list):
            indices = indices.tolist()
        fps = video_metadata["fps"]

        # Note this is a hardcode value for Qwen3-VL, should only be used for Qwen3-VL
        merge_size = 2
        if len(indices) % merge_size != 0:
            indices.extend(indices[-1] for _ in range(merge_size - len(indices) % merge_size))
        timestamps = [idx / fps for idx in indices]
        return timestamps

    def encode_image(self, image: Union[Image.Image, str]):
        if isinstance(image, str):
            img = Image.open(image).convert("RGB")
        else:
            img = image.copy()

        output_buffer = BytesIO()
        img.save(output_buffer, format="PNG")
        byte_data = output_buffer.getvalue()

        base64_str = base64.b64encode(byte_data).decode("utf-8")
        return base64_str
