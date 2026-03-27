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
        if video_kwargs is None:
            video_kwargs = {}

        openai_messages = []
        for message in self.messages:
            openai_message = {"role": message.role, "content": []}
            for content in message.content:
                if content.type == "text":
                    openai_message["content"].append({"type": "text", "text": content.text})
                elif content.type == "image":
                    openai_message["content"].append(
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{self.encode_image(content.url)}"},
                        }
                    )
                elif content.type == "video":
                    if fetch_video is None:
                        raise ImportError(
                            "qwen_vl_utils is required for video processing. Please install it with: pip install qwen-vl-utils"
                        )

                    algo = _resolve_algo(video_kwargs)
                    # req_fps = _resolve_fps(video_kwargs)
                    br = _resolve_budget_ratio(video_kwargs)

                    payload = {"type": "video", "video": content.url, **video_kwargs}

                    _apply_frame_and_patch_selection_to_payload(
                        video_url=content.url,
                        payload=payload,
                        video_kwargs=video_kwargs,
                        algo=algo,
                        budget_ratio=br,
                    )
                    video_input = fetch_video(payload)

                    frames_used = int(video_input.shape[0]) if hasattr(video_input, "shape") else len(video_input)
                    _record_frames_used(
                        video_path=content.url,
                        algo=algo,
                        fps=float(_BASE_FPS*br) if float(_BASE_FPS*br) is not None else -1.0,
                        frames_used=frames_used,
                    )

                    for frame in video_input:
                        image = Image.fromarray(frame.permute(1, 2, 0).numpy().astype(np.uint8))
                        openai_message["content"].append(
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{self.encode_image(image)}"},
                            }
                        )
                elif content.type == "audio":
                    openai_message["content"].append({"type": "audio_url", "audio_url": {"url": content.url}})

            openai_messages.append(openai_message)

        return openai_messages

    def to_qwen3_vl_openai_messages(self, video_kwargs: Dict[str, Any] = None):
        """
        Qwen3-VL: use fetch_video(..., return_video_metadata=True, return_video_sample_fps=True)
        and insert timestamps between frames.
        """
        if video_kwargs is None:
            video_kwargs = {}

        openai_messages = []
        for message in self.messages:
            openai_message = {"role": message.role, "content": []}
            for content in message.content:
                if content.type == "text":
                    openai_message["content"].append({"type": "text", "text": content.text})
                elif content.type == "image":
                    openai_message["content"].append(
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{self.encode_image(content.url)}"},
                        }
                    )
                elif content.type == "video":
                    if fetch_video is None:
                        raise ImportError(
                            "qwen_vl_utils is required for video processing. Please install it with: pip install qwen-vl-utils"
                        )

                    algo = _resolve_algo(video_kwargs)
                    # req_fps = _resolve_fps(video_kwargs)
                    br = _resolve_budget_ratio(video_kwargs)

                    payload = {"type": "video", "video": content.url, **video_kwargs}

                    _apply_frame_and_patch_selection_to_payload(
                        video_url=content.url,
                        payload=payload,
                        video_kwargs=video_kwargs,
                        algo=algo,
                        budget_ratio=br,
                    )
                    video_input, sampled_fps = fetch_video(
                        payload,
                        return_video_metadata=True,
                        return_video_sample_fps=True,
                    )

                    frames, video_metadata = video_input
                    frames_used = int(frames.shape[0])

                    # Prefer requested fps if set; else metadata fps (actual)
                    fps_for_log = float(_BASE_FPS*br) if float(_BASE_FPS*br) is not None else float(video_metadata.get("fps", -1.0))
                    _record_frames_used(
                        video_path=content.url,
                        algo=algo,
                        fps=fps_for_log,
                        frames_used=frames_used,
                    )

                    timestamps = self._calculate_timestamps(video_metadata)
                    for frame, timestamp in zip(frames, timestamps):
                        image = Image.fromarray(frame.permute(1, 2, 0).numpy().astype(np.uint8))
                        openai_message["content"].append({"type": "text", "text": f"<{timestamp:.1f} seconds>"})
                        openai_message["content"].append(
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{self.encode_image(image)}"},
                            }
                        )

                elif content.type == "audio":
                    openai_message["content"].append({"type": "audio_url", "audio_url": {"url": content.url}})

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
