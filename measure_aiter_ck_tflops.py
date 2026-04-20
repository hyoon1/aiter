#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
#
# Measure AITER CK FMHA forward TFLOPS with the same torch.cuda.Event timing style
# commonly used by FlashAttention microbenchmarks.
#
# Layout mapping against the C++ benchmark:
#   - --layout bshd ~= iperm=0
#   - --layout bhsd ~= iperm=1
#
# The Python dense API does not expose a separate operm toggle. It always returns
# a freshly allocated logical BSHD tensor. The varlen API returns packed [total, H, D].

import argparse
import csv
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from aiter.ops import mha as aiter_mha  # noqa: E402

SPECIAL_LENGTHS = [27280]


def estimate_flops(
    batch: int,
    nheads: int,
    seqlen_q: int,
    seqlen_k: int,
    head_dim_qk: int,
    head_dim_v: int,
    causal: bool,
) -> float:
    flops = batch * nheads * (
        2.0 * seqlen_q * seqlen_k * head_dim_qk
        + 2.0 * seqlen_q * seqlen_k * head_dim_v
    )
    return flops / 2.0 if causal else flops


def measure_with_repeats(run_kernel, burn_in: int, repeat: int) -> float:
    for _ in range(burn_in):
        run_kernel()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeat):
        run_kernel()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / repeat / 1000.0


def make_dense_tensor(
    batch: int,
    seqlen: int,
    nheads: int,
    head_dim: int,
    dtype: torch.dtype,
    layout: str,
) -> torch.Tensor:
    if layout == "bhsd":
        return torch.randn(
            batch, nheads, seqlen, head_dim, device="cuda", dtype=dtype
        ).permute(0, 2, 1, 3)
    return torch.randn(batch, seqlen, nheads, head_dim, device="cuda", dtype=dtype)


def make_varlen_tensor(
    total_tokens: int,
    nheads: int,
    head_dim: int,
    dtype: torch.dtype,
    layout: str,
) -> torch.Tensor:
    if layout == "bhsd":
        return torch.randn(
            nheads, total_tokens, head_dim, device="cuda", dtype=dtype
        ).permute(1, 0, 2)
    return torch.randn(total_tokens, nheads, head_dim, device="cuda", dtype=dtype)


def make_dense_runner(args, seqlen_q: int, seqlen_k: int, dtype: torch.dtype):
    q = make_dense_tensor(
        args.batch, seqlen_q, args.nheads, args.head_dim, dtype, args.layout
    )
    k = make_dense_tensor(
        args.batch, seqlen_k, args.nheads_k, args.head_dim, dtype, args.layout
    )
    v = make_dense_tensor(
        args.batch, seqlen_k, args.nheads_k, args.head_dim_v, dtype, args.layout
    )

    def run():
        return aiter_mha.flash_attn_func(
            q,
            k,
            v,
            dropout_p=0.0,
            softmax_scale=None,
            causal=args.causal,
            window_size=(-1, -1, 0),
            deterministic=False,
            return_lse=False,
            return_attn_probs=False,
        )

    stride_info = {
        "q": tuple(q.stride()),
        "k": tuple(k.stride()),
        "v": tuple(v.stride()),
    }
    return run, stride_info


def make_varlen_runner(args, seqlen_q: int, seqlen_k: int, dtype: torch.dtype):
    total_q = args.batch * seqlen_q
    total_k = args.batch * seqlen_k
    q = make_varlen_tensor(total_q, args.nheads, args.head_dim, dtype, args.layout)
    k = make_varlen_tensor(total_k, args.nheads_k, args.head_dim, dtype, args.layout)
    v = make_varlen_tensor(
        total_k, args.nheads_k, args.head_dim_v, dtype, args.layout
    )
    cu_q = torch.arange(
        0,
        total_q + 1,
        seqlen_q,
        device="cuda",
        dtype=torch.int32,
    )
    cu_k = torch.arange(
        0,
        total_k + 1,
        seqlen_k,
        device="cuda",
        dtype=torch.int32,
    )

    def run():
        return aiter_mha.flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q=cu_q,
            cu_seqlens_k=cu_k,
            max_seqlen_q=seqlen_q,
            max_seqlen_k=seqlen_k,
            min_seqlen_q=args.min_seqlen_q,
            dropout_p=0.0,
            softmax_scale=None,
            causal=args.causal,
            window_size=(-1, -1, 0),
            deterministic=False,
            return_lse=False,
            return_attn_probs=False,
        )

    stride_info = {
        "q": tuple(q.stride()),
        "k": tuple(k.stride()),
        "v": tuple(v.stride()),
    }
    return run, stride_info


def make_default_lengths(
    min_len: int,
    max_len: int,
    step: int,
    extra_lens: list[int] | None = None,
) -> list[int]:
    extra_lens = extra_lens or []
    lengths = [min_len]
    current = step
    while current <= max_len:
        if current not in lengths:
            lengths.append(current)
        current += step
    if lengths[-1] != max_len:
        lengths.append(max_len)
    for extra in extra_lens:
        if extra not in lengths:
            lengths.append(extra)
    return lengths


def parse_lengths(args) -> list[int]:
    if args.lengths:
        return [int(x.strip()) for x in args.lengths.split(",") if x.strip()]
    lengths = make_default_lengths(
        args.min_len,
        args.max_len,
        args.step,
        extra_lens=SPECIAL_LENGTHS,
    )
    if not lengths:
        raise ValueError("No lengths generated; check min/max/step.")
    return lengths


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def benchmark_mode(
    args,
    mode: str,
    lengths: list[int],
    dtype: torch.dtype,
) -> list[dict]:
    rows: list[dict] = []
    make_runner = make_dense_runner if mode == "dense" else make_varlen_runner
    first_stride_info = None

    print(f"--- {mode} ---")
    for seqlen_q in lengths:
        seqlen_k = args.seqlen_k if args.seqlen_k is not None else seqlen_q
        try:
            runner, stride_info = make_runner(args, seqlen_q, seqlen_k, dtype)
            if first_stride_info is None:
                first_stride_info = stride_info
            avg_s = measure_with_repeats(runner, args.burn_in, args.repeat)
            flops = estimate_flops(
                args.batch,
                args.nheads,
                seqlen_q,
                seqlen_k,
                args.head_dim,
                args.head_dim_v,
                args.causal,
            )
            tflops = flops / avg_s / 1.0e12
            row = {
                "mode": mode,
                "layout": args.layout,
                "batch": args.batch,
                "nheads_q": args.nheads,
                "nheads_kv": args.nheads_k,
                "head_dim_qk": args.head_dim,
                "head_dim_v": args.head_dim_v,
                "dtype": args.dtype,
                "causal": args.causal,
                "min_seqlen_q": args.min_seqlen_q,
                "seqlen_q": seqlen_q,
                "seqlen_k": seqlen_k,
                "time_ms": avg_s * 1.0e3,
                "tflops": tflops,
            }
            rows.append(row)
            print(
                f"Lq={seqlen_q:6d} Lk={seqlen_k:6d} "
                f"time={avg_s * 1.0e3:8.3f} ms TFLOPS={tflops:8.2f}"
            )
        except RuntimeError as e:
            print(
                f"Lq={seqlen_q:6d} Lk={seqlen_k:6d} failed: {type(e).__name__}: {e}"
            )
            torch.cuda.empty_cache()

    if args.print_strides and first_stride_info is not None:
        print(
            f"{mode} strides: q={first_stride_info['q']} "
            f"k={first_stride_info['k']} v={first_stride_info['v']}"
        )
    return rows


def parse_args():
    parser = argparse.ArgumentParser(
        description="Measure AITER CK FMHA forward TFLOPS with torch.cuda.Event timing."
    )
    parser.add_argument(
        "--mode",
        choices=["dense", "varlen", "both"],
        default="both",
        help="Which public Python op path to benchmark.",
    )
    parser.add_argument(
        "--layout",
        choices=["bshd", "bhsd"],
        default="bshd",
        help="Input physical layout. bshd ~= iperm=0, bhsd ~= iperm=1.",
    )
    parser.add_argument("--batch", type=int, default=1, help="Batch size.")
    parser.add_argument("--nheads", type=int, default=24, help="Q head count.")
    parser.add_argument(
        "--nheads-k",
        type=int,
        default=None,
        help="KV head count. Defaults to --nheads. Use a smaller value for GQA/MQA.",
    )
    parser.add_argument("--head-dim", type=int, default=128, help="Q/K head dim.")
    parser.add_argument(
        "--head-dim-v",
        type=int,
        default=None,
        help="V head dim. Defaults to --head-dim.",
    )
    parser.add_argument(
        "--dtype",
        choices=["fp16", "bf16"],
        default="bf16",
        help="Input dtype.",
    )
    parser.add_argument(
        "--lengths",
        type=str,
        default=None,
        help=(
            "Comma-separated query lengths. If set, overrides the default "
            "flash-attn-style sweep."
        ),
    )
    parser.add_argument("--min-len", type=int, default=1024, help="Minimum Lq.")
    parser.add_argument(
        "--max-len",
        type=int,
        default=28672,
        help="Maximum Lq in the base sweep; special lengths are appended after it.",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=4096,
        help="Base sweep step to match the flash-attn benchmark.",
    )
    parser.add_argument(
        "--seqlen-k",
        type=int,
        default=None,
        help="Fixed Lk. Defaults to Lk=Lq for each point.",
    )
    parser.add_argument(
        "--min-seqlen-q",
        type=int,
        default=0,
        help="Varlen min_seqlen_q. Keep 0 for the vLLM MM-style non-skip path.",
    )
    parser.add_argument(
        "--causal",
        action="store_true",
        help="Use causal attention. FLOP estimate is halved to match dense FA convention.",
    )
    parser.add_argument("--burn-in", type=int, default=5, help="Warmup iterations.")
    parser.add_argument("--repeat", type=int, default=20, help="Timed iterations.")
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=None,
        help="Optional CSV output path.",
    )
    parser.add_argument(
        "--print-strides",
        action="store_true",
        help="Print the first q/k/v strides to confirm BSHD vs BHSD.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA/ROCm device is required.")

    args.nheads_k = args.nheads if args.nheads_k is None else args.nheads_k
    args.head_dim_v = args.head_dim if args.head_dim_v is None else args.head_dim_v

    if args.nheads % args.nheads_k != 0:
        raise SystemExit("--nheads must be divisible by --nheads-k for GQA/MQA.")

    if not aiter_mha.ENABLE_CK:
        raise SystemExit("AITER CK path is not enabled in this environment.")

    lengths = parse_lengths(args)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    arch = aiter_mha.get_gfx()
    device_name = torch.cuda.get_device_name(torch.cuda.current_device())
    print(
        f"AITER CK benchmark arch={arch} device={device_name} "
        f"layout={args.layout} batch={args.batch} "
        f"hq={args.nheads} hk={args.nheads_k} "
        f"dqk={args.head_dim} dv={args.head_dim_v} dtype={args.dtype} "
        f"causal={args.causal} min_seqlen_q={args.min_seqlen_q}"
    )
    print(f"Lengths={lengths} Lk={'match Lq' if args.seqlen_k is None else args.seqlen_k}")

    all_rows: list[dict] = []
    modes = ["dense", "varlen"] if args.mode == "both" else [args.mode]
    for mode in modes:
        all_rows.extend(benchmark_mode(args, mode, lengths, dtype))

    if args.csv_path is not None:
        write_csv(args.csv_path, all_rows)
        print(f"Wrote CSV to {args.csv_path}")


if __name__ == "__main__":
    main()
