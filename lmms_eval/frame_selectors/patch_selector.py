from __future__ import annotations

import json
import math
import os
import subprocess
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

try:
    import cv_reader.api as cv_api  # type: ignore
    _HAS_CV_READER = True
except Exception:
    cv_api = None
    _HAS_CV_READER = False


@dataclass
class PatchSelectorConfig:
    patch_size: int = 16
    square_size: int = 576
    keep_ratio: float = 0.125
    num_patches_per_frame: Optional[int] = None
    iframe_full: bool = True
    mv_unit_div: float = 4.0
    mv_pct: float = 95.0
    res_pct: float = 95.0
    mv_compensate: str = "median"
    res_use_grad: bool = False
    fuse_mode: str = "weighted"
    w_mv: float = 1.0
    w_res: float = 1.0
    sort_within_frame: bool = True
    return_debug: bool = False


def _run_ffprobe_pict_types(video_path: str) -> List[str]:
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_frames",
        "-show_entries", "frame=pict_type",
        "-of", "csv=p=0",
        video_path,
    ]
    out = subprocess.check_output(cmd, text=True)
    return [x.strip() for x in out.splitlines() if x.strip()]


def _ensure_cv_reader() -> None:
    if not _HAS_CV_READER:
        raise RuntimeError("cv_reader is not available. Install Compressed_Video_Reader first.")


def _read_all_compressed_frames(video_path: str) -> List[Dict[str, Any]]:
    _ensure_cv_reader()
    frames = cv_api.read_video(video_path, 0, -1)
    if not isinstance(frames, (list, tuple)) or len(frames) == 0:
        raise RuntimeError(f"cv_reader.read_video returned empty result for: {video_path}")
    return list(frames)


def _residual_y_generic(residual: np.ndarray) -> np.ndarray:
    x = np.asarray(residual)
    if x.ndim == 2:
        return x.astype(np.float32)
    if x.ndim != 3:
        raise ValueError(f"Unexpected residual ndim={x.ndim}, shape={x.shape}")
    if x.shape[-1] == 1:
        return x[..., 0].astype(np.float32)
    ch0 = x[..., 0].astype(np.float32)
    ch1 = x[..., 1].astype(np.float32)
    ch2 = x[..., 2].astype(np.float32)
    return (0.114 * ch0 + 0.587 * ch1 + 0.299 * ch2).astype(np.float32)


def _percentile_alpha(x: np.ndarray, pct: float) -> float:
    finite = x[np.isfinite(x)]
    if finite.size == 0:
        return 1.0
    a = float(np.percentile(finite, pct))
    return max(a, 1e-6)


def _residual_energy_norm(residual_y: np.ndarray, pct: float = 95.0, use_grad: bool = False) -> Tuple[np.ndarray, float]:
    y = residual_y.astype(np.float32)
    if use_grad:
        gx = cv2.Sobel(y, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(y, cv2.CV_32F, 0, 1, ksize=3)
        e = np.sqrt(gx * gx + gy * gy)
    else:
        e = np.abs(y - 128.0)
    alpha = _percentile_alpha(e, pct)
    e = np.clip(e / alpha, 0.0, 1.0).astype(np.float32)
    return e, alpha


def _mv_energy_norm(mv_x: np.ndarray, mv_y: np.ndarray, H: int, W: int, mv_unit_div: float = 4.0, pct: float = 95.0, compensate: str = "median") -> Tuple[np.ndarray, float]:
    vx = np.asarray(mv_x).astype(np.float32) / float(mv_unit_div)
    vy = np.asarray(mv_y).astype(np.float32) / float(mv_unit_div)
    if compensate == "median":
        vx = vx - np.median(vx)
        vy = vy - np.median(vy)
    elif compensate == "mean":
        vx = vx - np.mean(vx)
        vy = vy - np.mean(vy)
    elif compensate != "none":
        raise ValueError(f"Unsupported mv_compensate: {compensate}")
    mag = np.sqrt(vx * vx + vy * vy).astype(np.float32)
    alpha = _percentile_alpha(mag, pct)
    mag = np.clip(mag / alpha, 0.0, 1.0)
    mag_full = cv2.resize(mag, (W, H), interpolation=cv2.INTER_NEAREST).astype(np.float32)
    return mag_full, alpha


def _fuse_energy(mv_norm: np.ndarray, res_norm: np.ndarray, mode: str = "weighted", w_mv: float = 1.0, w_res: float = 1.0) -> np.ndarray:
    if mode == "weighted":
        denom = max(w_mv + w_res, 1e-6)
        out = (w_mv * mv_norm + w_res * res_norm) / denom
    elif mode == "sum":
        out = w_mv * mv_norm + w_res * res_norm
    elif mode == "max":
        out = np.maximum(w_mv * mv_norm, w_res * res_norm)
    elif mode == "geomean":
        out = np.sqrt(np.maximum(w_mv * mv_norm, 0.0) * np.maximum(w_res * res_norm, 0.0))
    else:
        raise ValueError(f"Unsupported fuse_mode: {mode}")
    return out.astype(np.float32)


def _resize_longer_pad_square_map(x: np.ndarray, out_size: int = 576) -> Tuple[np.ndarray, Dict[str, Any]]:
    H, W = x.shape[:2]
    scale = float(out_size) / float(max(H, W))
    Hn = max(1, int(round(H * scale)))
    Wn = max(1, int(round(W * scale)))
    x_rs = cv2.resize(x.astype(np.float32), (Wn, Hn), interpolation=cv2.INTER_LINEAR)
    pad_top = (out_size - Hn) // 2
    pad_left = (out_size - Wn) // 2
    pad_bottom = out_size - Hn - pad_top
    pad_right = out_size - Wn - pad_left
    out = np.pad(x_rs, ((pad_top, pad_bottom), (pad_left, pad_right)), mode="constant", constant_values=0.0).astype(np.float32)
    pad_info = {"H0": H, "W0": W, "scale": scale, "Hn": Hn, "Wn": Wn, "pad_top": pad_top, "pad_left": pad_left, "pad_bottom": pad_bottom, "pad_right": pad_right, "out_size": out_size}
    return out, pad_info


def _make_valid_patch_mask_from_padinfo(pad_info: Dict[str, Any], square_size: int, patch_size: int) -> np.ndarray:
    valid = np.zeros((square_size, square_size), dtype=np.uint8)
    y0 = int(pad_info["pad_top"]); x0 = int(pad_info["pad_left"]); Hn = int(pad_info["Hn"]); Wn = int(pad_info["Wn"])
    valid[y0:y0+Hn, x0:x0+Wn] = 1
    hb = square_size // patch_size; wb = square_size // patch_size
    patch_valid = (valid.reshape(hb, patch_size, wb, patch_size).sum(axis=(1, 3)) > 0)
    return patch_valid.astype(bool)


def _patch_sum_map(x_sq: np.ndarray, patch_size: int) -> np.ndarray:
    hb = x_sq.shape[0] // patch_size; wb = x_sq.shape[1] // patch_size
    return x_sq.reshape(hb, patch_size, wb, patch_size).sum(axis=(1, 3)).astype(np.float32)


def _frame_patch_budget(valid_patch_count: int, keep_ratio: float, num_patches_per_frame: Optional[int], iframe_full: bool, is_iframe: bool) -> int:
    if iframe_full and is_iframe:
        return int(valid_patch_count)
    if num_patches_per_frame is not None:
        return int(max(1, min(num_patches_per_frame, valid_patch_count)))
    k = int(math.floor(float(keep_ratio) * float(valid_patch_count)))
    return max(1, min(k, valid_patch_count))


def _select_topk_patch_indices(patch_scores: np.ndarray, valid_mask: np.ndarray, k: int, sort_desc: bool = True) -> np.ndarray:
    flat_scores = patch_scores.reshape(-1)
    valid_idx = np.flatnonzero(valid_mask.reshape(-1))
    if valid_idx.size == 0:
        return np.zeros((0, 2), dtype=np.int64)
    k = min(k, valid_idx.size)
    vals = flat_scores[valid_idx]
    if k == valid_idx.size:
        chosen = valid_idx.copy()
    else:
        local_top = np.argpartition(-vals, k - 1)[:k]
        chosen = valid_idx[local_top]
    if sort_desc:
        chosen = chosen[np.argsort(-flat_scores[chosen])]
    wb = patch_scores.shape[1]
    hs = chosen // wb; ws = chosen % wb
    return np.stack([hs, ws], axis=1).astype(np.int64)


def _frame_pict_type(frame: Dict[str, Any]) -> str:
    return str(frame.get("pict_type", "")).strip().upper()


def _compute_frame_patch_selection(frame: Dict[str, Any], local_t: int, cfg: PatchSelectorConfig, pict_type_fallback: Optional[str] = None) -> Dict[str, Any]:
    H0 = int(frame["height"]); W0 = int(frame["width"])
    pict_type = _frame_pict_type(frame) or (str(pict_type_fallback).strip().upper() if pict_type_fallback else "")
    is_iframe = (pict_type == "I")
    if cfg.iframe_full and is_iframe:
        dummy = np.ones((cfg.square_size, cfg.square_size), dtype=np.float32)
        _, pad_info = _resize_longer_pad_square_map(dummy, out_size=cfg.square_size)
        valid_mask = _make_valid_patch_mask_from_padinfo(pad_info, cfg.square_size, cfg.patch_size)
        all_hw = np.argwhere(valid_mask)
        if cfg.sort_within_frame:
            all_hw = all_hw[np.lexsort((all_hw[:, 1], all_hw[:, 0]))]
        thw = np.concatenate([np.full((all_hw.shape[0], 1), local_t, dtype=np.int64), all_hw.astype(np.int64)], axis=1)
        return {"pict_type": pict_type, "is_iframe": True, "num_selected": int(thw.shape[0]), "patch_positions": thw, "debug": None}

    mv = np.asarray(frame["motion_vector"]); res = np.asarray(frame["residual"])
    residual_y = _residual_y_generic(res)
    res_norm, res_alpha = _residual_energy_norm(residual_y, pct=cfg.res_pct, use_grad=cfg.res_use_grad)
    mv_norm, mv_alpha = _mv_energy_norm(mv[..., 0], mv[..., 1], H=H0, W=W0, mv_unit_div=cfg.mv_unit_div, pct=cfg.mv_pct, compensate=cfg.mv_compensate)
    fused_hw = _fuse_energy(mv_norm, res_norm, mode=cfg.fuse_mode, w_mv=cfg.w_mv, w_res=cfg.w_res)
    fused_sq, pad_info = _resize_longer_pad_square_map(fused_hw, out_size=cfg.square_size)
    valid_mask = _make_valid_patch_mask_from_padinfo(pad_info, cfg.square_size, cfg.patch_size)
    patch_scores = _patch_sum_map(fused_sq, patch_size=cfg.patch_size)
    valid_patch_count = int(valid_mask.sum())
    k = _frame_patch_budget(valid_patch_count, cfg.keep_ratio, cfg.num_patches_per_frame, cfg.iframe_full, is_iframe)
    hw = _select_topk_patch_indices(patch_scores, valid_mask, k, sort_desc=cfg.sort_within_frame)
    thw = np.concatenate([np.full((hw.shape[0], 1), local_t, dtype=np.int64), hw.astype(np.int64)], axis=1)
    debug = None
    if cfg.return_debug:
        debug = {"mv_alpha": float(mv_alpha), "res_alpha": float(res_alpha), "pad_info": pad_info, "valid_patch_count": valid_patch_count, "patch_scores_shape": list(patch_scores.shape)}
    return {"pict_type": pict_type, "is_iframe": is_iframe, "num_selected": int(thw.shape[0]), "patch_positions": thw, "debug": debug}


def select_patch_positions_for_frames(video_path: str, selected_frame_indices: Sequence[int], cfg: Optional[PatchSelectorConfig] = None) -> Dict[str, Any]:
    if cfg is None:
        cfg = PatchSelectorConfig()
    frames_all = _read_all_compressed_frames(video_path)
    total_num_frames = len(frames_all)
    selected = [int(x) for x in selected_frame_indices]
    if len(selected) == 0:
        return {"patch_positions": np.zeros((0, 3), dtype=np.int64), "frame_types": [], "num_patches_per_frame": [], "selected_frame_indices": [], "config": asdict(cfg), "debug": [] if cfg.return_debug else None}
    for idx in selected:
        if idx < 0 or idx >= total_num_frames:
            raise IndexError(f"selected_frame_indices contains out-of-range index {idx} (total_num_frames={total_num_frames})")
    fallback_types = None
    if any(_frame_pict_type(frames_all[idx]) == "" for idx in selected):
        try:
            fallback_types = _run_ffprobe_pict_types(video_path)
        except Exception:
            fallback_types = None

    all_thw, frame_types, num_patches_per_frame, debug_list = [], [], [], []
    for local_t, global_idx in enumerate(selected):
        fr = frames_all[global_idx]
        fallback_pt = fallback_types[global_idx] if (fallback_types is not None and global_idx < len(fallback_types)) else None
        out = _compute_frame_patch_selection(fr, local_t, cfg, fallback_pt)
        all_thw.append(out["patch_positions"])
        frame_types.append(str(out["pict_type"]))
        num_patches_per_frame.append(int(out["num_selected"]))
        if cfg.return_debug:
            debug_list.append({"global_frame_index": int(global_idx), "local_t": int(local_t), "pict_type": str(out["pict_type"]), "num_selected": int(out["num_selected"]), "debug": out["debug"]})

    patch_positions = np.concatenate(all_thw, axis=0).astype(np.int64) if len(all_thw) > 0 else np.zeros((0, 3), dtype=np.int64)
    return {"patch_positions": patch_positions, "frame_types": frame_types, "num_patches_per_frame": num_patches_per_frame, "selected_frame_indices": selected, "config": asdict(cfg), "debug": debug_list if cfg.return_debug else None}


def save_patch_selector_output(out: Dict[str, Any], npy_path: str, json_path: Optional[str] = None) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(npy_path)), exist_ok=True)
    np.save(npy_path, out["patch_positions"])
    if json_path is not None:
        meta = {k: v for k, v in out.items() if k != "patch_positions"}
        os.makedirs(os.path.dirname(os.path.abspath(json_path)), exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)


__all__ = ["PatchSelectorConfig", "select_patch_positions_for_frames", "save_patch_selector_output"]
