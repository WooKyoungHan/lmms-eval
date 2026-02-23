from __future__ import annotations

import json
import math
import subprocess
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

import hashlib
import os
from pathlib import Path

_IPB_META_CACHE_DIR = os.environ.get(
    "IPB_META_CACHE_DIR",
    os.path.expanduser("~/.cache/ipb_meta")
)

def _video_fingerprint(path: str) -> str:
    st = os.stat(path)
    h = hashlib.sha1()
    h.update(str(st.st_size).encode())
    h.update(str(st.st_mtime_ns).encode())

    with open(path, "rb") as f:
        h.update(f.read(1024 * 1024))
    return h.hexdigest()

def _meta_cache_path(video_path: str) -> Path:
    Path(_IPB_META_CACHE_DIR).mkdir(parents=True, exist_ok=True)
    key = _video_fingerprint(video_path)
    return Path(_IPB_META_CACHE_DIR) / f"{key}.json"

# -----------------------------
# ffprobe helpers (no decode)
# -----------------------------
def _ffprobe_pict_types_and_pkt_sizes(video_path: str):
    cache_file = _meta_cache_path(video_path)

    # 캐시가 있는데 비어있으면(이전 실패 캐시) 무시하고 재생성
    if cache_file.exists():
        data = json.loads(cache_file.read_text())
        if data.get("ipb") and data.get("sizes"):
            return data["ipb"], data["sizes"]

    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_frames",
        "-show_entries", "frame=pkt_size,pict_type",
        "-of", "csv",
        video_path,
    ]

    out = subprocess.check_output(cmd).decode("utf-8", errors="ignore")

    ipb: list[str] = []
    sizes: list[int] = []

    for line in out.splitlines():
        # 빈 줄 방지
        line = line.strip()
        if not line:
            continue

        parts = [x.strip() for x in line.split(",")]

        # 최소: frame, <pkt_size>, <pict_field>
        if len(parts) < 3 or parts[0] != "frame":
            continue

        pkt = parts[1]
        pict_field = parts[2]  # "B" / "P" / "I" / "Iside_data" 등

        # pict는 첫 글자만
        pict = pict_field[:1]
        if pict not in ("I", "P", "B"):
            continue

        # pkt_size 숫자 파싱
        try:
            size = int(pkt)
        except ValueError:
            continue

        ipb.append(pict)
        sizes.append(size)

    if not ipb:
        raise RuntimeError("ffprobe parsing produced 0 frames (unexpected).")

    cache_file.write_text(json.dumps({"ipb": ipb, "sizes": sizes}))
    return ipb, sizes

def _ffprobe_fps(video_path: str) -> Optional[float]:
    """
    Best-effort FPS read via ffprobe (no decode).
    average_frame_rate / r_frame_rate 중 우선.
    """
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=avg_frame_rate,r_frame_rate",
        "-of",
        "json",
        video_path,
    ]
    try:
        out = subprocess.check_output(cmd).decode("utf-8", errors="ignore")
        data = json.loads(out)
        st = (data.get("streams") or [{}])[0]

        def _to_float(frac: str) -> Optional[float]:
            if not frac or frac == "0/0":
                return None
            if "/" in frac:
                a, b = frac.split("/", 1)
                a = float(a)
                b = float(b)
                if b == 0:
                    return None
                return a / b
            return float(frac)

        fps = _to_float(st.get("avg_frame_rate") or "")
        if fps is not None and fps > 0:
            return fps
        fps = _to_float(st.get("r_frame_rate") or "")
        if fps is not None and fps > 0:
            return fps
    except Exception:
        return None
    return None


def _pict_to_code(p: str) -> int:
    # sorting_algorithm의 매핑과 동일 :contentReference[oaicite:2]{index=2}
    return {"I": 1, "P": 0, "B": -1}.get(p, 0)


def _get_ipb_indices(ipb_code: torch.Tensor) -> Tuple[List[int], List[int], List[int]]:
    ipb_cpu = ipb_code.detach().cpu()
    I_idx = (ipb_cpu == 1).nonzero(as_tuple=False).squeeze(1).tolist()
    P_idx = (ipb_cpu == 0).nonzero(as_tuple=False).squeeze(1).tolist()
    B_idx = (ipb_cpu == -1).nonzero(as_tuple=False).squeeze(1).tolist()
    return I_idx, P_idx, B_idx


# -----------------------------
# policy knobs
# -----------------------------
@dataclass(frozen=True)
class IPBSelectorConfig:
    # budget control
    fps: Optional[float] = None          # e.g., 0.5, 0.25
    num_frm_cap: int = 10000               # absolute upper bound

    # adaptive policy (from model_utils.py) :contentReference[oaicite:3]{index=3}
    beta_cov: float = 0.6
    seg_sec: float = 2.0
    nms_radius_sec: float = 0.25
    g_ref: float = 0.5
    include_all_I: bool = True

    # scoring
    mode_global: str = "bytes"           # "bytes" or "typed_bytes"
    type_weights: Optional[Dict[str, float]] = None
    priority_for_cov_rep: Sequence[str] = ("P", "B", "I")

    # fallback FPS if probing fails
    fallback_src_fps: float = 30.0


# -----------------------------
# core helpers (index-only)
# -----------------------------
def _compute_budget_K(T: int, src_fps: float, target_fps: Optional[float], num_frm_cap: int) -> int:
    # model_utils의 K 계산 로직을 index-only로 복제 :contentReference[oaicite:4]{index=4}
    if T <= 0:
        return 0
    if target_fps is None or target_fps <= 0 or src_fps <= 0:
        return min(T, int(num_frm_cap))
    duration = T / float(src_fps)
    K = int(round(duration * float(target_fps)))
    K = max(1, K)
    return min(T, K, int(num_frm_cap))


def _temporal_nms_pick(
    candidates: List[int],
    scores: torch.Tensor,
    k: int,
    *,
    src_fps: float,
    nms_radius_sec: float,
    already_selected: Optional[set] = None,
) -> List[int]:
    # model_utils의 temporal NMS를 index-only로 복제 :contentReference[oaicite:5]{index=5}
    if k <= 0 or not candidates:
        return []
    already_selected = already_selected or set()

    rad = max(0, int(round(nms_radius_sec * float(src_fps))))
    cand_sorted = sorted(candidates, key=lambda i: float(scores[i]), reverse=True)

    if rad <= 0:
        out = []
        for i in cand_sorted:
            if i in already_selected:
                continue
            out.append(i)
            if len(out) >= k:
                break
        return out

    chosen: List[int] = []
    blocked = set()

    for s in already_selected:
        for j in range(s - rad, s + rad + 1):
            blocked.add(j)

    for i in cand_sorted:
        if i in already_selected or i in blocked:
            continue
        chosen.append(i)
        for j in range(i - rad, i + rad + 1):
            blocked.add(j)
        if len(chosen) >= k:
            break
    return chosen


def _choose_coverage_segments(
    *,
    T: int,
    src_fps: float,
    I_idx: Sequence[int],
    K_cov: int,
    seg_sec: float,
) -> List[Tuple[int, int]]:
    # model_utils의 coverage segment 선택을 index-only로 복제 :contentReference[oaicite:6]{index=6}
    if K_cov <= 0:
        return []

    seg_len = max(1, int(round(seg_sec * float(src_fps))))
    n_seg = int(math.ceil(T / seg_len))

    seg_has_I = np.zeros((n_seg,), dtype=np.bool_)
    for i in I_idx:
        s = min(n_seg - 1, i // seg_len)
        seg_has_I[s] = True

    cand = [s for s in range(n_seg) if not seg_has_I[s]]
    if not cand:
        return []

    I_segs = np.where(seg_has_I)[0].tolist()
    if not I_segs:
        take = cand[:K_cov]
        return [(s * seg_len, min(T, (s + 1) * seg_len)) for s in take]

    def dist_to_nearest_I(seg: int) -> int:
        return min(abs(seg - si) for si in I_segs)

    cand_sorted = sorted(cand, key=lambda s: dist_to_nearest_I(s), reverse=True)
    take = cand_sorted[:K_cov]
    return [(s * seg_len, min(T, (s + 1) * seg_len)) for s in take]


# -----------------------------
# public API
# -----------------------------
def select_frame_indices_ipb(
    video_path: str,
    cfg: IPBSelectorConfig,
) -> List[int]:
    """
    Returns: sorted frame indices (display order 기준)
    - ffprobe로 ipb + bytes만 읽고
    - adaptive policy로 indices만 선택한다.
    """
    ipb_str, pkt_sizes = _ffprobe_pict_types_and_pkt_sizes(video_path)
    T = min(len(ipb_str), len(pkt_sizes))
    if T <= 0:
        return []

    ipb_code = torch.tensor([_pict_to_code(p) for p in ipb_str[:T]], dtype=torch.int8)
    frame_bytes = torch.tensor(pkt_sizes[:T], dtype=torch.int64)

    src_fps = _ffprobe_fps(video_path)
    if src_fps is None or src_fps <= 0:
        src_fps = float(cfg.fallback_src_fps)

    K = _compute_budget_K(T, float(src_fps), cfg.fps, int(cfg.num_frm_cap))
    if K <= 0:
        return []

    I_idx, P_idx, B_idx = _get_ipb_indices(ipb_code)
    I_set, P_set, B_set = set(I_idx), set(P_idx), set(B_idx)

    fb = frame_bytes.to(torch.float32)

    # global scores
    if cfg.mode_global == "bytes":
        scores = fb
    elif cfg.mode_global == "typed_bytes":
        w = cfg.type_weights or {"I": 1.0, "P": 1.0, "B": 1.0}

        def code_to_typ(c: int) -> str:
            if c == 1:
                return "I"
            if c == 0:
                return "P"
            if c == -1:
                return "B"
            return "P"

        scores = torch.zeros((T,), dtype=torch.float32)
        for i in range(T):
            typ = code_to_typ(int(ipb_code[i].item()))
            scores[i] = fb[i] * float(w.get(typ, 1.0))
    else:
        raise ValueError(f"Unknown mode_global: {cfg.mode_global}")

    chosen_set: set[int] = set()
    if cfg.include_all_I:
        chosen_set |= I_set

    # include_all_I인데 I만으로 K 초과 시: top-K만 남김 (model_utils 로직) :contentReference[oaicite:7]{index=7}
    if cfg.include_all_I and len(chosen_set) > K:
        chosen = sorted(list(chosen_set), key=lambda i: float(scores[i]), reverse=True)[:K]
        return sorted(chosen)

    # regime score g from I_per_sec :contentReference[oaicite:8]{index=8}
    duration_sec = T / float(src_fps)
    I_per_sec = (len(I_idx) / max(duration_sec, 1e-6))
    g = float(I_per_sec / max(cfg.g_ref, 1e-6))
    g = max(0.0, min(1.0, g))

    K_extra = max(0, K - len(chosen_set))
    if K_extra <= 0:
        return sorted(list(chosen_set))

    # allocate coverage vs global :contentReference[oaicite:9]{index=9}
    K_cov = int(round((1.0 - g) * float(cfg.beta_cov) * K_extra))
    K_cov = max(0, min(K_cov, K_extra))
    K_glb = K_extra - K_cov

    # coverage picks :contentReference[oaicite:10]{index=10}
    if K_cov > 0:
        segs = _choose_coverage_segments(
            T=T, src_fps=float(src_fps), I_idx=I_idx, K_cov=K_cov, seg_sec=float(cfg.seg_sec)
        )

        for (a, b) in segs:
            idxs = list(range(a, b))
            if not idxs:
                continue

            def typ_rank(i: int) -> int:
                # priority_for_cov_rep: ("P","B","I") 기본
                # P 우선, 그 다음 B, 마지막 I
                if i in P_set:
                    return 0 if "P" in cfg.priority_for_cov_rep else 2
                if i in B_set:
                    return 1 if "B" in cfg.priority_for_cov_rep else 2
                if i in I_set:
                    return 2
                return 3

            idxs_sorted = sorted(idxs, key=lambda i: (typ_rank(i), -float(scores[i])))
            for i in idxs_sorted:
                if i not in chosen_set:
                    chosen_set.add(i)
                    break

    # global picks with temporal NMS :contentReference[oaicite:11]{index=11}
    if K_glb > 0:
        candidates = [i for i in range(T) if i not in chosen_set]
        glb_picks = _temporal_nms_pick(
            candidates,
            scores,
            K_glb,
            src_fps=float(src_fps),
            nms_radius_sec=float(cfg.nms_radius_sec),
            already_selected=chosen_set,
        )
        chosen_set |= set(glb_picks)

    chosen = sorted(list(chosen_set))

    # hard cap always enforced (model_utils의 최종 cap) :contentReference[oaicite:12]{index=12}
    if len(chosen) > K:
        chosen = sorted(chosen, key=lambda i: float(scores[i]), reverse=True)[:K]
        chosen = sorted(chosen)

    return chosen


# Convenience: minimal constructor for your common use
def build_default_cfg(fps: Optional[float]) -> IPBSelectorConfig:
    return IPBSelectorConfig(fps=fps)
