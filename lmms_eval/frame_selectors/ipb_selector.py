from __future__ import annotations

import json
import math
import subprocess
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

import hashlib
import os
from pathlib import Path

import heapq

try:
    import av  # type: ignore
except Exception:
    av = None

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

def _meta_cache_path_with_suffix(video_path: str, suffix: str) -> Path:
    Path(_IPB_META_CACHE_DIR).mkdir(parents=True, exist_ok=True)
    key = _video_fingerprint(video_path)
    return Path(_IPB_META_CACHE_DIR) / f"{key}_{suffix}.json"

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
# @dataclass(frozen=True)
@dataclass(frozen=True)
class IPBSelectorConfig:
    # budget control
    fps: Optional[float] = None
    budget_ratio: float = 1.0
    max_budget: int = 2048
    num_frm_cap: int = 10000
    budget_frames: Optional[int] = None
    # adaptive policy (existing)
    beta_cov: float = 0.6
    seg_sec: float = 2.0
    nms_radius_sec: float = 0.25
    hybrid_uniform_ratio: float = 0.0
    hybrid_min_dist_sec: Optional[float] = None
    hybrid_max_refill_rounds: int = 8
    g_ref: float = 0.5
    include_all_I: bool = True

    # scoring
    mode_global: str = "bytes"
    type_weights: Optional[Dict[str, float]] = None
    priority_for_cov_rep: Sequence[str] = ("P", "B", "I")

    # fallback FPS if probing fails
    fallback_src_fps: float = 30.0

    # new: GOP proportional-fair allocator
    propfair_min_per_gop_if_possible: bool = True

    # 
    utility: str = "log"
    alpha: float = 2.0
    beta: float = 0.5

    # version 3: motion-aware GOP utility weight
    # utility_v3 = log(1 + normalized pkt/frame) * (1 - exp(-beta * normalized motion/frame))
    v3_motion_beta: float = 2.0
    v3_enable_motion_cache: bool = True

    # suspicious GOP detection is used only to exclude GOPs from the
    # "minimum 1 frame per GOP" floor allocation.
    # Default: high pkt/frame + low motion efficiency = suspicious.
    v3_susp_rate_q: float = 0.85
    v3_susp_eff_q: float = 0.20
# -----------------------------
# core helpers (index-only)
# -----------------------------
def _compute_budget_K(
    T: int,
    src_fps: float,
    target_fps: Optional[float],
    budget_ratio: float,
    max_budget: int,
    num_frm_cap: int,
    budget_frames: Optional[int] = None,
) -> int:
    if T <= 0:
        return 0

    if budget_frames is not None:
        K = int(budget_frames)
        K = max(1, K)
        return min(T, K, int(num_frm_cap))

    br = float(budget_ratio)
    if br < 0.0 or br > 1.0:
        raise ValueError(f"budget_ratio must be in [0, 1], got {budget_ratio}")
    if br == 0.0:
        return 0

    base_budget = float(max_budget)
    if target_fps is not None and target_fps > 0 and src_fps > 0:
        duration_sec = T / float(src_fps)
        base_budget = min(duration_sec * float(target_fps), float(max_budget))

    K = int(round(base_budget * br))
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


def _pick_uniform_indices(T: int, k: int) -> List[int]:
    if T <= 0 or k <= 0:
        return []
    k = min(int(k), int(T))
    idx = ((np.arange(k, dtype=np.float64) + 0.5) * float(T) / float(k)).astype(np.int64)
    idx = np.clip(idx, 0, T - 1)
    out: List[int] = []
    seen = set()
    for i in idx.tolist():
        ii = int(i)
        if ii not in seen:
            out.append(ii)
            seen.add(ii)
    if len(out) < k:
        for i in range(T):
            if i not in seen:
                out.append(i)
                seen.add(i)
            if len(out) >= k:
                break
    return sorted(out[:k])


def _resolve_min_dist_radius_sec(cfg: IPBSelectorConfig) -> float:
    if cfg.hybrid_min_dist_sec is None:
        return float(cfg.nms_radius_sec)
    return float(cfg.hybrid_min_dist_sec)


def _build_blocked_mask(T: int, selected: Sequence[int], src_fps: float, radius_sec: float) -> np.ndarray:
    blocked = np.zeros((T,), dtype=np.bool_)
    if T <= 0:
        return blocked
    rad = max(0, int(round(float(radius_sec) * float(src_fps))))
    for s in selected:
        a = max(0, int(s) - rad)
        b = min(T, int(s) + rad + 1)
        blocked[a:b] = True
    return blocked


def _count_available_per_gop(
    gops: Sequence[Tuple[int, int]],
    blocked_mask: np.ndarray,
) -> List[int]:
    caps: List[int] = []
    for a, b in gops:
        caps.append(int((~blocked_mask[a:b]).sum()))
    return caps


def _pick_topk_in_segment_allowed(
    *,
    a: int,
    b: int,
    k: int,
    scores: torch.Tensor,
    frame_bytes: torch.Tensor,
    ipb_code: torch.Tensor,
    allowed_mask: np.ndarray,
) -> List[int]:
    if k <= 0 or b <= a:
        return []

    candidates = [i for i in range(a, b) if bool(allowed_mask[i])]
    if not candidates:
        return []

    k = min(int(k), len(candidates))
    if k == 1:
        i_idx = [i for i in candidates if int(ipb_code[i].item()) == 1]
        if i_idx:
            best = max(i_idx, key=lambda i: int(frame_bytes[i].item()))
            return [best]
        best = max(candidates, key=lambda i: float(scores[i]))
        return [best]

    if k >= len(candidates):
        return sorted(candidates)

    pos = ((np.arange(k, dtype=np.float64) + 0.5) * float(len(candidates)) / float(k)).astype(np.int64)
    pos = np.clip(pos, 0, len(candidates) - 1)
    picked = sorted({candidates[int(j)] for j in pos.tolist()})
    if len(picked) < k:
        used = set(picked)
        leftovers = [i for i in candidates if i not in used]
        picked.extend(leftovers[: (k - len(picked))])
    return sorted(picked[:k])


def _allocate_and_pick_from_available(
    *,
    gops: Sequence[Tuple[int, int]],
    weights: torch.Tensor,
    scores: torch.Tensor,
    frame_bytes: torch.Tensor,
    ipb_code: torch.Tensor,
    blocked_mask: np.ndarray,
    K_target: int,
    cfg: IPBSelectorConfig,
    suspicious_mask: Optional[np.ndarray],
    alloc_utility: str,
) -> List[int]:
    if K_target <= 0 or not gops:
        return []

    caps = _count_available_per_gop(gops, blocked_mask)
    total_avail = int(sum(caps))
    if total_avail <= 0:
        return []

    K_eff = min(int(K_target), total_avail)
    budgets = _allocate_gop_budgets_heap(
        weights,
        caps,
        K_eff,
        utility=alloc_utility,
        alpha=float(cfg.alpha),
        beta=float(cfg.beta),
        min_per_gop_if_possible=cfg.propfair_min_per_gop_if_possible,
        min_floor_mask=None if suspicious_mask is None else (~suspicious_mask).tolist(),
    )

    allowed_mask = ~blocked_mask
    chosen: List[int] = []
    for gi, (a, b) in enumerate(gops):
        k_i = int(budgets[gi])
        if k_i <= 0:
            continue
        picks = _pick_topk_in_segment_allowed(
            a=a,
            b=b,
            k=k_i,
            scores=scores,
            frame_bytes=frame_bytes,
            ipb_code=ipb_code,
            allowed_mask=allowed_mask,
        )
        chosen.extend(picks)
    return sorted(set(chosen))


def _hybrid_refill_selection(
    *,
    T: int,
    gops: Sequence[Tuple[int, int]],
    weights: torch.Tensor,
    scores: torch.Tensor,
    frame_bytes: torch.Tensor,
    ipb_code: torch.Tensor,
    initial_selected: Sequence[int],
    src_fps: float,
    cfg: IPBSelectorConfig,
    K: int,
    suspicious_mask: Optional[np.ndarray],
    alloc_utility: str,
) -> List[int]:
    chosen = sorted(set(int(i) for i in initial_selected))
    radius_sec = _resolve_min_dist_radius_sec(cfg)

    for _ in range(max(1, int(cfg.hybrid_max_refill_rounds))):
        if len(chosen) >= K:
            break
        blocked_mask = _build_blocked_mask(T, chosen, float(src_fps), radius_sec)
        new_picks = _allocate_and_pick_from_available(
            gops=gops,
            weights=weights,
            scores=scores,
            frame_bytes=frame_bytes,
            ipb_code=ipb_code,
            blocked_mask=blocked_mask,
            K_target=(K - len(chosen)),
            cfg=cfg,
            suspicious_mask=suspicious_mask,
            alloc_utility=alloc_utility,
        )
        new_only = [i for i in new_picks if i not in set(chosen)]
        if not new_only:
            break
        chosen = sorted(set(chosen).union(new_only))

    if len(chosen) < K:
        blocked_mask = _build_blocked_mask(T, chosen, float(src_fps), radius_sec)
        leftovers = [i for i in range(T) if not bool(blocked_mask[i]) and i not in set(chosen)]
        leftovers = sorted(leftovers, key=lambda i: float(scores[i]), reverse=True)
        chosen.extend(leftovers[: (K - len(chosen))])
        chosen = sorted(set(chosen))

    return chosen[:K]


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

    K = _compute_budget_K(
        T,
        float(src_fps),
        cfg.fps,
        float(cfg.budget_ratio),
        int(cfg.max_budget),
        int(cfg.num_frm_cap),
        budget_frames=cfg.budget_frames,
    )
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


def _build_frame_scores(
    *,
    ipb_code: torch.Tensor,
    frame_bytes: torch.Tensor,
    mode_global: str,
    type_weights: Optional[Dict[str, float]],
) -> torch.Tensor:
    fb = frame_bytes.to(torch.float32)

    if mode_global == "bytes":
        return fb

    if mode_global == "typed_bytes":
        w = type_weights or {"I": 1.0, "P": 1.0, "B": 1.0}
        scores = torch.zeros((len(ipb_code),), dtype=torch.float32)
        for i in range(len(ipb_code)):
            c = int(ipb_code[i].item())
            if c == 1:
                typ = "I"
            elif c == 0:
                typ = "P"
            elif c == -1:
                typ = "B"
            else:
                typ = "P"
            scores[i] = fb[i] * float(w.get(typ, 1.0))
        return scores

    raise ValueError(f"Unknown mode_global: {mode_global}")


def _build_gop_ranges_from_I(ipb_code: torch.Tensor) -> List[Tuple[int, int]]:
    """
    GOP 정의: I부터 다음 I 직전까지.
    frame 0이 I가 아니어도 [0, first_I) prefix를 하나의 segment로 처리.
    """
    T = int(ipb_code.numel())
    if T <= 0:
        return []

    starts = [0]
    for i in range(1, T):
        if int(ipb_code[i].item()) == 1:  # I-frame
            starts.append(i)

    gops: List[Tuple[int, int]] = []
    for j, a in enumerate(starts):
        b = starts[j + 1] if (j + 1) < len(starts) else T
        if b > a:
            gops.append((a, b))
    return gops


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _frame_motion_mass_from_rows(rows: Sequence[Sequence[float]]) -> float:
    total = 0.0
    for row in rows:
        if len(row) < 9:
            continue
        w = float(row[4])
        h = float(row[5])
        scale = float(row[8])
        if scale == 0.0:
            scale = 1.0
        dx = float(row[6]) / scale
        dy = float(row[7]) / scale
        total += (w * h) * math.sqrt(dx * dx + dy * dy)
    return total


def _motion_vectors_side_data(frame: Any):
    side_data = getattr(frame, "side_data", None)
    if side_data is None:
        return None

    try:
        mv_sd = side_data.get("MOTION_VECTORS")
        if mv_sd is not None:
            return mv_sd
    except Exception:
        pass

    try:
        for sd in side_data:
            sd_type = str(getattr(sd, "type", ""))
            if "MOTION_VECTORS" in sd_type or "Motion vectors" in sd_type:
                return sd
    except Exception:
        pass
    return None


def _extract_frame_motion_masses_pyav(video_path: str, *, use_cache: bool = True) -> List[float]:
    cache_file = _meta_cache_path_with_suffix(video_path, "motion_mass")
    if use_cache and cache_file.exists():
        try:
            data = json.loads(cache_file.read_text())
            vals = data.get("motion_mass", [])
            if vals:
                return [float(v) for v in vals]
        except Exception:
            pass

    if av is None:
        raise RuntimeError("PyAV is not available; version 3 motion-aware utility requires av.")

    container = av.open(video_path)
    stream = container.streams.video[0]

    try:
        stream.codec_context.flags2 |= av.codec.context.Flags2.EXPORT_MVS
    except Exception:
        try:
            stream.codec_context.options = {"flags2": "+export_mvs"}
        except Exception:
            pass

    out: List[float] = []
    for packet in container.demux(stream):
        for frame in packet.decode():
            mv_sd = _motion_vectors_side_data(frame)
            if mv_sd is None:
                out.append(0.0)
                continue

            rows = []
            try:
                iterator = list(mv_sd)
            except Exception:
                iterator = []

            for mv in iterator:
                rows.append([
                    float(getattr(mv, "src_x", 0) or 0),
                    float(getattr(mv, "src_y", 0) or 0),
                    float(getattr(mv, "dst_x", 0) or 0),
                    float(getattr(mv, "dst_y", 0) or 0),
                    float(getattr(mv, "w", 0) or 0),
                    float(getattr(mv, "h", 0) or 0),
                    float(getattr(mv, "motion_x", 0) or 0),
                    float(getattr(mv, "motion_y", 0) or 0),
                    float(getattr(mv, "motion_scale", 1) or 1),
                ])
            out.append(_frame_motion_mass_from_rows(rows))

    container.close()

    if use_cache and out:
        try:
            cache_file.write_text(json.dumps({"motion_mass": out}))
        except Exception:
            pass
    return out


def _compute_gop_stats_v3(
    gops: Sequence[Tuple[int, int]],
    frame_scores: torch.Tensor,
    frame_motion_mass: Sequence[float],
    src_fps: float,
) -> List[Dict[str, float]]:
    stats: List[Dict[str, float]] = []
    motion_np = np.asarray(frame_motion_mass, dtype=np.float64)
    for gi, (a, b) in enumerate(gops):
        L = max(1, int(b - a))
        duration = L / max(float(src_fps), 1e-6)
        pkt_sum = float(frame_scores[a:b].sum().item())
        pkt_per_frame = pkt_sum / max(float(L), 1e-9)
        motion_sum = float(motion_np[a:b].sum()) if motion_np.size > 0 else 0.0
        motion_per_frame = motion_sum / max(float(L), 1e-9)
        motion_efficiency = motion_per_frame / max(pkt_per_frame, 1e-12)
        stats.append({
            "gop_id": float(gi),
            "start": float(a),
            "end": float(b),
            "num_frames": float(L),
            "duration": float(duration),
            "pkt_sum": float(pkt_sum),
            "pkt_per_frame": float(pkt_per_frame),
            "motion_mass_sum": float(motion_sum),
            "motion_per_frame": float(motion_per_frame),
            "motion_efficiency": float(motion_efficiency),
        })
    return stats


def _compute_gop_propfair_weights_v3(
    gops: Sequence[Tuple[int, int]],
    frame_scores: torch.Tensor,
    frame_motion_mass: Sequence[float],
    src_fps: float,
    *,
    motion_beta: float,
) -> Tuple[torch.Tensor, List[Dict[str, float]]]:
    if not gops:
        return torch.zeros((0,), dtype=torch.float32), []

    stats = _compute_gop_stats_v3(gops, frame_scores, frame_motion_mass, src_fps)

    R = np.asarray([s["pkt_per_frame"] for s in stats], dtype=np.float64)
    M = np.asarray([s["motion_per_frame"] for s in stats], dtype=np.float64)

    R_ref = float(np.median(R[R > 0])) if np.any(R > 0) else 1.0
    M_ref = float(np.median(M[M > 0])) if np.any(M > 0) else 1.0

    weights = torch.zeros((len(gops),), dtype=torch.float32)
    for gi, s in enumerate(stats):
        Rn = s["pkt_per_frame"] / max(R_ref, 1e-12)
        Mn = s["motion_per_frame"] / max(M_ref, 1e-12)

        bit_term = math.log1p(Rn)
        motion_gate = 1.0 - math.exp(-float(motion_beta) * Mn)
        utility_v3 = bit_term * motion_gate

        s["Rn"] = float(Rn)
        s["Mn"] = float(Mn)
        s["bit_term"] = float(bit_term)
        s["motion_gate"] = float(motion_gate)
        s["utility_v3"] = float(utility_v3)
        s["R_ref"] = float(R_ref)
        s["M_ref"] = float(M_ref)
        weights[gi] = max(float(utility_v3), 1e-6)

    return weights, stats


def _compute_suspicious_gop_mask_v3(
    stats: Sequence[Dict[str, float]],
    *,
    rate_q: float,
    eff_q: float,
) -> np.ndarray:
    """
    Suspicious GOPs are excluded only from the minimum-per-GOP floor allocation.

    Criterion:
        high pkt/frame  AND  low motion_efficiency
    where
        motion_efficiency = motion_per_frame / max(pkt_per_frame, eps)
    """
    if not stats:
        return np.zeros((0,), dtype=bool)

    rate = np.asarray([float(s.get("pkt_per_frame", 0.0)) for s in stats], dtype=np.float64)
    eff = np.asarray([float(s.get("motion_efficiency", 0.0)) for s in stats], dtype=np.float64)

    rate_thr = float(np.quantile(rate, rate_q)) if rate.size > 0 else 0.0
    eff_thr = float(np.quantile(eff, eff_q)) if eff.size > 0 else 0.0

    mask = np.zeros((len(stats),), dtype=bool)
    for i, s in enumerate(stats):
        is_suspicious = bool(
            float(s.get("pkt_per_frame", 0.0)) >= rate_thr
            and float(s.get("motion_efficiency", 0.0)) <= eff_thr
        )
        s["suspicious_rate_thr"] = rate_thr
        s["suspicious_eff_thr"] = eff_thr
        s["is_suspicious"] = is_suspicious
        mask[i] = is_suspicious
    return mask


def _compute_gop_propfair_weights(
    gops: Sequence[Tuple[int, int]],
    frame_scores: torch.Tensor,
) -> torch.Tensor:
    """
    1) 가장 기본형: proportional-fair utility
       u_i(b_i) = w_i * log(1 + b_i)

    여기서 w_i는 GOP 내부 frame-level score의 평균으로 둔다.
    mode_global="bytes"면 w_i = mean(pkt_size) = S_i / L_i.
    (fps가 GOP마다 동일하므로 S_i/T_i와 비례)
    """
    if not gops:
        return torch.zeros((0,), dtype=torch.float32)

    w = torch.zeros((len(gops),), dtype=torch.float32)
    for gi, (a, b) in enumerate(gops):
        seg = frame_scores[a:b]
        if seg.numel() <= 0:
            w[gi] = 1e-6
        else:
            w[gi] = max(float(seg.mean().item()), 1e-6)
    return w


def _allocate_gop_budgets_propfair(
    weights: torch.Tensor,
    caps: Sequence[int],
    K: int,
    *,
    min_per_gop_if_possible: bool,
    min_floor_mask: Optional[Sequence[bool]] = None,
    utility: str = "log",   # "log" | "alpha_fair" | "exp"
    alpha: float = 2.0,            # for alpha_fair (>=1, 1이면 log)
    beta: float = 0.5,             # for exp
) -> List[int]:
    """
    Discrete proportional-fair allocation:
        maximize sum_i w_i * log(1 + b_i)
        s.t. sum_i b_i = K, 0 <= b_i <= cap_i, b_i integer

    separable concave integer allocation이므로,
    marginal gain greedy로 푸는 형태.
    """
    G = int(weights.numel())
    if G <= 0 or K <= 0:
        return [0] * G

    w = weights.detach().cpu().tolist()
    b = [0] * G
    remaining = int(K)

    floor_mask = [True] * G if min_floor_mask is None else [bool(x) for x in min_floor_mask]
    n_floor = sum(1 for i in range(G) if floor_mask[i] and caps[i] > 0)

    # 가능하면 GOP당 1장 floor를 먼저 깐다 (coverage 쪽)
    if min_per_gop_if_possible and K >= n_floor:
        for i in range(G):
            if floor_mask[i] and caps[i] > 0:
                b[i] = 1
                remaining -= 1

    for _ in range(max(0, remaining)):
        best_i = -1
        best_delta = -1.0

        for i in range(G):
            if b[i] >= caps[i]:
                continue

            # u_i(b_i+1) - u_i(b_i)
            if utility == "log":
                delta = w[i] * (math.log1p(b[i] + 1) - math.log1p(b[i]))

            elif utility == "alpha_fair":
                if abs(alpha - 1.0) < 1e-9:
                    delta = w[i] * (math.log1p(b[i] + 1) - math.log1p(b[i]))
                else:
                    delta = w[i] * ((b[i] + 2) ** (1 - alpha) - (b[i] + 1) ** (1 - alpha))

            elif utility == "exp":
                delta = w[i] * (math.exp(-beta * b[i]) - math.exp(-beta * (b[i] + 1)))

            else:
                raise ValueError(f"unknown utility: {utility}")
            if delta > best_delta:
                best_delta = delta
                best_i = i

        if best_i < 0:
            break
        b[best_i] += 1

    return b



def _delta(utility: str, w: float, b: int, alpha: float, beta: float) -> float:
    if utility == "log":
        return w * (math.log1p(b + 1) - math.log1p(b))
    if utility == "alpha_fair":
        if abs(alpha - 1.0) < 1e-9:
            return w * (math.log1p(b + 1) - math.log1p(b))
        return w * ((b + 2) ** (1 - alpha) - (b + 1) ** (1 - alpha))
    if utility == "exp":
        return w * (math.exp(-beta * b) - math.exp(-beta * (b + 1)))
    raise ValueError(f"unknown utility: {utility}")

def _allocate_gop_budgets_heap(
    weights: torch.Tensor,
    caps: Sequence[int],
    K: int,
    *,
    min_per_gop_if_possible: bool,
    min_floor_mask: Optional[Sequence[bool]] = None,
    utility: str = "log",
    alpha: float = 2.0,
    beta: float = 0.5,
) -> List[int]:
    G = int(weights.numel())
    if G <= 0 or K <= 0:
        return [0] * G

    w = weights.detach().cpu().tolist()
    b = [0] * G
    remaining = int(K)

    floor_mask = [True] * G if min_floor_mask is None else [bool(x) for x in min_floor_mask]
    n_floor = sum(1 for i in range(G) if floor_mask[i] and caps[i] > 0)

    # 1) optional floor
    if min_per_gop_if_possible and K >= n_floor:
        for i in range(G):
            if floor_mask[i] and caps[i] > 0:
                b[i] = 1
                remaining -= 1

    # 2) heap init with current deltas
    heap = []
    for i in range(G):
        if b[i] < caps[i]:
            d = _delta(utility, w[i], b[i], alpha, beta)
            # max-heap via negative
            heapq.heappush(heap, (-d, i))

    # 3) greedy K steps, update only chosen i
    while remaining > 0 and heap:
        negd, i = heapq.heappop(heap)
        if b[i] >= caps[i]:
            continue  # stale
        # apply one unit
        b[i] += 1
        remaining -= 1

        # push updated delta for same i
        if b[i] < caps[i]:
            d = _delta(utility, w[i], b[i], alpha, beta)
            heapq.heappush(heap, (-d, i))

    return b

def _pick_topk_in_segment(
    *,
    a: int,
    b: int,
    k: int,
    scores: torch.Tensor,
    frame_bytes: torch.Tensor,
    ipb_code: torch.Tensor,
    src_fps: float,
    nms_radius_sec: float,
) -> List[int]:
    """
    segment [a, b) 안에서 실제 frame idx를 선택한다.

    - k == 1:
        I-frame만 선택 (GOP 내 I가 없으면 score 기준 top-1 fallback)
    - k >= 2:
        GOP 내부에서 P -> B -> I 순서로,
        각 타입 내부는 pkt_size(frame_bytes) 내림차순으로 선택
    """
    if k <= 0 or b <= a:
        return []

    candidates = list(range(a, b))
    k = min(k, len(candidates))

    if k == 1:
        i_idx = [i for i in candidates if int(ipb_code[i].item()) == 1]
        if len(i_idx) > 0:
            # I-frame only: choose the largest packet among I-frames in this GOP
            best = max(i_idx, key=lambda i: int(frame_bytes[i].item()))
            return [best]
        # Fallback: if this segment has no I-frame (e.g., prefix before first I), use score top-1
        best = max(candidates, key=lambda i: float(scores[i]))
        return [best]

    if k >= 2:
        i_idx = [i for i in candidates if int(ipb_code[i].item()) == 1]
        p_idx = [i for i in candidates if int(ipb_code[i].item()) == 0]
        b_idx = [i for i in candidates if int(ipb_code[i].item()) == -1]

        i_sorted = sorted(i_idx, key=lambda i: int(frame_bytes[i].item()), reverse=True)
        p_sorted = sorted(p_idx, key=lambda i: int(frame_bytes[i].item()), reverse=True)
        b_sorted = sorted(b_idx, key=lambda i: int(frame_bytes[i].item()), reverse=True)

        ordered = i_sorted +p_sorted + b_sorted 
        return sorted(ordered[:k])
    # if k >= 2:
    #     L = b - a
    #     if k >= L:
    #         return list(range(a, b))

    #     # uniform sampling (segment midpoints)
    #     idx = a + ((np.arange(k) + 0.5) * L / k).astype(int)
    #     return idx.tolist()
    picked = _temporal_nms_pick(
        candidates,
        scores,
        k,
        src_fps=float(src_fps),
        nms_radius_sec=float(nms_radius_sec),
        already_selected=None,
    )

    if len(picked) < k:
        used = set(picked)
        leftovers = sorted(
            [i for i in candidates if i not in used],
            key=lambda i: float(scores[i]),
            reverse=True,
        )
        picked.extend(leftovers[: (k - len(picked))])

    return sorted(picked)

def _choose_L_min(gops, src_fps, q=5.0, t_min_sec=0.4, hard_cap=None):
    lens = np.array([b-a for (a,b) in gops], dtype=np.int32)
    if lens.size == 0:
        return 0

    Lq = int(np.percentile(lens, q))
    Lt = int(np.floor(float(src_fps) * float(t_min_sec)))

    L_min = max(Lq, Lt)

    # 너무 커져서 과도 삭제 방지용 (선택)
    if hard_cap is not None:
        L_min = min(L_min, int(hard_cap))

    return max(1, L_min)

def select_frame_indices_ipb_propfair_gop(
    video_path: str,
    cfg: IPBSelectorConfig,
) -> List[int]:
    """
    GOP 단위 proportional-fair allocation + optional hybrid uniform anchors.

    1) 전체 budget K 중 일부를 uniform anchor로 먼저 고정
    2) 남은 budget은 GOP-based allocator로 분배
    3) 이미 선택된 frame 근처(중복/근접)는 block 처리
    4) 부족분이 생기면 남은 가용 frame에 대해 allocator를 다시 돌려 재분배
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

    K = _compute_budget_K(
        T,
        float(src_fps),
        cfg.fps,
        float(cfg.budget_ratio),
        int(cfg.max_budget),
        int(cfg.num_frm_cap),
        budget_frames=cfg.budget_frames,
    )
    if K <= 0:
        return []

    scores = _build_frame_scores(
        ipb_code=ipb_code,
        frame_bytes=frame_bytes,
        mode_global=cfg.mode_global,
        type_weights=cfg.type_weights,
    )

    gops_raw = _build_gop_ranges_from_I(ipb_code)
    L_min = _choose_L_min(gops_raw, src_fps, q=10.0, t_min_sec=1)
    gops = [(a, b) for (a, b) in gops_raw if (b - a) >= L_min]
    if not gops:
        return []

    suspicious_mask: Optional[np.ndarray] = None
    alloc_utility = cfg.utility
    if cfg.utility == "v3":
        frame_motion_mass = _extract_frame_motion_masses_pyav(
            video_path,
            use_cache=bool(cfg.v3_enable_motion_cache),
        )
        Tm = min(T, len(frame_motion_mass))
        if Tm < T:
            frame_motion_mass = list(frame_motion_mass) + [0.0] * (T - Tm)
        else:
            frame_motion_mass = list(frame_motion_mass[:T])

        weights, gop_debug_stats = _compute_gop_propfair_weights_v3(
            gops,
            scores,
            frame_motion_mass,
            float(src_fps),
            motion_beta=float(cfg.v3_motion_beta),
        )
        suspicious_mask = _compute_suspicious_gop_mask_v3(
            gop_debug_stats,
            rate_q=float(cfg.v3_susp_rate_q),
            eff_q=float(cfg.v3_susp_eff_q),
        )
        alloc_utility = "exp"
    else:
        weights = _compute_gop_propfair_weights(gops, scores)

    uniform_ratio = float(cfg.hybrid_uniform_ratio)
    if uniform_ratio < 0.0 or uniform_ratio > 1.0:
        raise ValueError(f"hybrid_uniform_ratio must be in [0, 1], got {uniform_ratio}")

    K_uniform = min(K, max(0, int(round(K * uniform_ratio))))
    uniform_selected = _pick_uniform_indices(T, K_uniform)

    blocked_mask = _build_blocked_mask(T, uniform_selected, float(src_fps), _resolve_min_dist_radius_sec(cfg))
    adaptive_selected = _allocate_and_pick_from_available(
        gops=gops,
        weights=weights,
        scores=scores,
        frame_bytes=frame_bytes,
        ipb_code=ipb_code,
        blocked_mask=blocked_mask,
        K_target=(K - len(uniform_selected)),
        cfg=cfg,
        suspicious_mask=suspicious_mask,
        alloc_utility=alloc_utility,
    )

    chosen = sorted(set(uniform_selected).union(adaptive_selected))
    chosen = _hybrid_refill_selection(
        T=T,
        gops=gops,
        weights=weights,
        scores=scores,
        frame_bytes=frame_bytes,
        ipb_code=ipb_code,
        initial_selected=chosen,
        src_fps=float(src_fps),
        cfg=cfg,
        K=K,
        suspicious_mask=suspicious_mask,
        alloc_utility=alloc_utility,
    )

    if len(chosen) > K:
        chosen = sorted(chosen, key=lambda i: float(scores[i]), reverse=True)[:K]
        chosen = sorted(chosen)
    return chosen

def analyze_ipb_propfair_gop(
    video_path: str,
    cfg: IPBSelectorConfig,
) -> Dict[str, Any]:
    """
    Notebook/debug helper.
    Returns GOP-wise stats, weights, budgets, and chosen indices.
    For utility="v3", GOP stats include the motion-aware terms.
    """
    ipb_str, pkt_sizes = _ffprobe_pict_types_and_pkt_sizes(video_path)
    T = min(len(ipb_str), len(pkt_sizes))
    if T <= 0:
        return {
            "T": 0,
            "gops": [],
            "weights": torch.zeros((0,), dtype=torch.float32),
            "budgets": [],
            "selected": [],
            "gop_stats": [],
        }

    ipb_code = torch.tensor([_pict_to_code(p) for p in ipb_str[:T]], dtype=torch.int8)
    frame_bytes = torch.tensor(pkt_sizes[:T], dtype=torch.int64)
    scores = _build_frame_scores(
        ipb_code=ipb_code,
        frame_bytes=frame_bytes,
        mode_global=cfg.mode_global,
        type_weights=cfg.type_weights,
    )

    src_fps = _ffprobe_fps(video_path)
    if src_fps is None or src_fps <= 0:
        src_fps = float(cfg.fallback_src_fps)

    K = _compute_budget_K(
        T,
        float(src_fps),
        cfg.fps,
        float(cfg.budget_ratio),
        int(cfg.max_budget),
        int(cfg.num_frm_cap),
        budget_frames=cfg.budget_frames,
    )

    gops_raw = _build_gop_ranges_from_I(ipb_code)
    L_min = _choose_L_min(gops_raw, src_fps, q=10.0, t_min_sec=1)
    gops = [(a, b) for (a, b) in gops_raw if (b - a) >= L_min]

    alloc_utility = cfg.utility
    suspicious_mask: Optional[np.ndarray] = None
    if cfg.utility == "v3":
        frame_motion_mass = _extract_frame_motion_masses_pyav(
            video_path,
            use_cache=bool(cfg.v3_enable_motion_cache),
        )
        Tm = min(T, len(frame_motion_mass))
        if Tm < T:
            frame_motion_mass = list(frame_motion_mass) + [0.0] * (T - Tm)
        else:
            frame_motion_mass = list(frame_motion_mass[:T])
        weights, gop_stats = _compute_gop_propfair_weights_v3(
            gops,
            scores,
            frame_motion_mass,
            float(src_fps),
            motion_beta=float(cfg.v3_motion_beta),
        )
        suspicious_mask = _compute_suspicious_gop_mask_v3(
            gop_stats,
            rate_q=float(cfg.v3_susp_rate_q),
            eff_q=float(cfg.v3_susp_eff_q),
        )
        alloc_utility = "exp"
    else:
        weights = _compute_gop_propfair_weights(gops, scores)
        gop_stats = []
        for gi, (a, b) in enumerate(gops):
            gop_stats.append({
                "gop_id": float(gi),
                "start": float(a),
                "end": float(b),
                "num_frames": float(b - a),
                "weight": float(weights[gi].item()),
            })

    caps = [b - a for (a, b) in gops]
    budgets = _allocate_gop_budgets_heap(
        weights,
        caps,
        K,
        utility=alloc_utility,
        alpha=float(cfg.alpha),
        beta=float(cfg.beta),
        min_per_gop_if_possible=cfg.propfair_min_per_gop_if_possible,
        min_floor_mask=None if suspicious_mask is None else (~suspicious_mask).tolist(),
    )

    for gi in range(len(gops)):
        if gi < len(gop_stats):
            gop_stats[gi]["weight"] = float(weights[gi].item())
            gop_stats[gi]["budget"] = int(budgets[gi])

    selected = select_frame_indices_ipb_propfair_gop(video_path, cfg)

    return {
        "T": T,
        "src_fps": float(src_fps),
        "K": int(K),
        "gops": gops,
        "weights": weights,
        "budgets": budgets,
        "scores": scores,
        "selected": selected,
        "gop_stats": gop_stats,
        "L_min": int(L_min),
    }
