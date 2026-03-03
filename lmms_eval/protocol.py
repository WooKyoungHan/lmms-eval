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

# Optional video processing dependencies
VideoReader, _has_decord = optional_import("decord", "VideoReader")
cpu, _ = optional_import("decord", "cpu")
fetch_video, _has_qwen_vl = optional_import("qwen_vl_utils", "fetch_video")

_FRAME_STATS = {"records": []}
_BASE_FPS = 2.0

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
        if v in {"ipb", "ipb_v2"}:
            return v
        return "default"

    env = os.environ.get("LMMS_USE_ALGO", "").strip().lower()
    if env in {"ipb", "ipb_v2", "default"}:
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
    """
    Decide IPB utility function for GOP allocator.
    Priority:
      1) video_kwargs["utility"] if present
      2) env LMMS_IPB_UTILITY
      3) "log"
    """
    v = (video_kwargs or {}).get("utility", None)
    if isinstance(v, str) and v.strip():
        v = v.strip().lower()
        if v in {"log", "alpha_fair", "exp"}:
            return v

    env = os.environ.get("LMMS_IPB_UTILITY", "").strip().lower()
    if env in {"log", "alpha_fair", "exp"}:
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
        encode_cache: Dict[Tuple[object, ...], str] = {}
        image_format = os.getenv("LMMS_IMAGE_ENCODE_FORMAT", "PNG").upper()
        mime_type = f"image/{'jpeg' if image_format == 'JPG' else image_format.lower()}"
        quality = int(os.getenv("LMMS_IMAGE_JPEG_QUALITY", "85")) if image_format in {"JPEG", "JPG", "WEBP"} else None
        for message in self.messages:
            openai_message = {"role": message.role, "content": []}
            for content in message.content:
                if content.type == "text":
                    openai_message["content"].append({"type": "text", "text": content.text})
                elif content.type == "image":
                    openai_message["content"].append(
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime_type};base64,{self.encode_image(content.url, encode_cache, image_format, quality)}"},
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

                    # Ensure the resolved fps is actually present in payload if decided from env
                    if algo == "default":
                        payload["fps"] = float(_BASE_FPS*br)                     # If algo is ipb / ipb_v2, add frame_indices (pre-decode)
                    elif algo in {"ipb", "ipb_v2"}:
                        utility = _resolve_utility(video_kwargs)
                        alpha, beta = _resolve_alpha_beta(video_kwargs)

                        # cfg = IPBSelectorConfig(fps=req_fps, utility=utility, alpha=alpha, beta=beta)
                        cfg = IPBSelectorConfig(fps=float(_BASE_FPS), budget_ratio=br, utility=utility, alpha=alpha, beta=beta)
                        if algo == "ipb":
                            payload["frame_indices"] = select_frame_indices_ipb(content.url, cfg)
                        else:
                            payload["frame_indices"] = select_frame_indices_ipb_propfair_gop(content.url, cfg)
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
                                "image_url": {"url": f"data:{mime_type};base64,{self.encode_image(image, encode_cache, image_format, quality)}"},
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
        encode_cache: Dict[Tuple[object, ...], str] = {}
        image_format = os.getenv("LMMS_IMAGE_ENCODE_FORMAT", "PNG").upper()
        mime_type = f"image/{'jpeg' if image_format == 'JPG' else image_format.lower()}"
        quality = int(os.getenv("LMMS_IMAGE_JPEG_QUALITY", "85")) if image_format in {"JPEG", "JPG", "WEBP"} else None
        for message in self.messages:
            openai_message = {"role": message.role, "content": []}
            for content in message.content:
                if content.type == "text":
                    openai_message["content"].append({"type": "text", "text": content.text})
                elif content.type == "image":
                    openai_message["content"].append(
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime_type};base64,{self.encode_image(content.url, encode_cache, image_format, quality)}"},
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

                    # Ensure the resolved fps is actually present in payload if decided from env
                    if algo == "default":
                        payload["fps"] = float(_BASE_FPS*br)   

                    elif algo in {"ipb", "ipb_v2"}:
                        utility = _resolve_utility(video_kwargs)
                        alpha, beta = _resolve_alpha_beta(video_kwargs)
                        cfg = IPBSelectorConfig(fps=float(_BASE_FPS), utility=utility, alpha=alpha, beta=beta)
                        if algo == "ipb":
                            payload["frame_indices"] = select_frame_indices_ipb(content.url, cfg)
                        else:
                            payload["frame_indices"] = select_frame_indices_ipb_propfair_gop(content.url, cfg)
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
                                "image_url": {"url": f"data:{mime_type};base64,{self.encode_image(image, encode_cache, image_format, quality)}"},
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

    def encode_image(
        self,
        image: Union[Image.Image, str],
        cache: Optional[Dict[Tuple[object, ...], str]] = None,
        image_format: str = "PNG",
        quality: Optional[int] = None,
    ):
        normalized_image_format = image_format.upper()
        return encode_image_to_base64(
            image,
            image_format=normalized_image_format,
            convert_rgb=normalized_image_format in {"JPEG", "JPG", "WEBP"},
            quality=quality,
            copy_if_pil=False,
            cache=cache,
            use_path_cache=True,
        )
