"""Benchmark Hamming LSBM embedding and simulation."""

from __future__ import annotations

import argparse
import time
import numpy as np
import conseal as cl


def _safe_div(num: float, den: float) -> float:
    if den == 0:
        return float('inf')
    return num / den

def _bench_true(
    k: int,
    num_blocks: int,
    rng_cover: np.random.Generator,
    rng_msg: np.random.Generator,
    rng_embed: np.random.Generator,
) -> tuple[int, int, float, float]:
    n = 2**k - 1
    cover = rng_cover.integers(0, 256, size=(num_blocks, n), dtype=np.uint8)
    msg = rng_msg.integers(0, 2, size=(num_blocks, k), dtype=np.uint8)

    changes = 0
    start = time.perf_counter()
    for i in range(num_blocks):
        stego = cl.coding.hamming.embed_lsbm_hamming(
            cover[i],
            msg[i],
            k,
            rng=rng_embed
        )
        changes += int(np.count_nonzero(cover[i] != stego))
    elapsed = time.perf_counter() - start

    e_true = _safe_div(k * num_blocks, changes)
    return n, changes, e_true, elapsed


def _bench_sim_k(
    k: int,
    num_blocks: int,
    rng_sim: np.random.Generator,
) -> tuple[int, float, float]:
    start = time.perf_counter()
    changes = cl.coding.hamming.simulate_embedding(k, num_blocks, rng=rng_sim)
    elapsed = time.perf_counter() - start
    e_sim = _safe_div(k * num_blocks, changes)
    return changes, e_sim, elapsed


def _bench_sim_alpha(
    alpha: float,
    num_elements: int,
    rng_sim: np.random.Generator,
) -> tuple[int, float, float]:
    start = time.perf_counter()
    changes = cl.coding.hamming.simulate_embedding_alpha(
        alpha,
        num_elements,
        rng=rng_sim
    )
    elapsed = time.perf_counter() - start
    e_sim = _safe_div(alpha * num_elements, changes)
    return changes, e_sim, elapsed


def _print_k_results(rows: list[dict]) -> None:
    print("\nTrue vs simulated embedding (Hamming k)")
    print(
        "k  n  alpha   e_true  e_sim  |e_true-e_sim|  "
        "t_true_ms  t_sim_ms  speedup"
    )
    for row in rows:
        speedup = _safe_div(row["time_true"], row["time_sim"]) if row["time_sim"] > 0 else float('inf')
        print(
            f"{row['k']:>2} {row['n']:>2} {row['alpha']:.4f} "
            f"{row['e_true']:.3f} {row['e_sim']:.3f} "
            f"{abs(row['e_true'] - row['e_sim']):.3f} "
            f"{row['time_true'] * 1e3:>9.3f} {row['time_sim'] * 1e3:>8.3f} "
            f"{speedup:>7.2f}"
        )


def _print_alpha_results(rows: list[dict]) -> None:
    print("\nAlpha interpolation and detectability (e_uncoded=2)")
    print("alpha  e_analytical  e_sim  t_sim_ms  detectability_gain")
    for row in rows:
        gain = _safe_div(row["e_analytical"], 2.0)
        print(
            f"{row['alpha']:.4f}  {row['e_analytical']:.3f} "
            f"{row['e_sim']:.3f} {row['time_sim'] * 1e3:>8.3f} "
            f"{gain:>7.3f}"
        )


def run() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark Hamming LSBM embedding and simulation."
    )
    parser.add_argument(
        "--k",
        type=int,
        nargs="*",
        default=[2, 3, 4],
        help="Hamming code parameters k to benchmark."
    )
    parser.add_argument(
        "--num-blocks",
        type=int,
        default=1000,
        help="Number of blocks for true embedding and k-based simulation."
    )
    parser.add_argument(
        "--alphas",
        type=float,
        nargs="*",
        default=[0.4, 0.5, 0.8],
        help="Embedding rates to benchmark with interpolation."
    )
    parser.add_argument(
        "--num-elements",
        type=int,
        default=100000,
        help="Number of elements for alpha-based simulation."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=12345,
        help="Random seed for reproducibility."
    )
    args = parser.parse_args()

    seed_seq = np.random.SeedSequence(args.seed)
    rngs = [np.random.default_rng(s) for s in seed_seq.spawn(4)]
    rng_cover, rng_msg, rng_embed, rng_sim = rngs

    k_rows = []
    for k in args.k:
        n, changes_true, e_true, time_true = _bench_true(
            k,
            args.num_blocks,
            rng_cover,
            rng_msg,
            rng_embed
        )
        changes_sim, e_sim, time_sim = _bench_sim_k(
            k,
            args.num_blocks,
            rng_sim
        )
        alpha = k / n
        k_rows.append({
            "k": k,
            "n": n,
            "alpha": alpha,
            "changes_true": changes_true,
            "changes_sim": changes_sim,
            "e_true": e_true,
            "e_sim": e_sim,
            "time_true": time_true,
            "time_sim": time_sim,
        })

    _print_k_results(k_rows)

    alpha_rows = []
    for alpha in args.alphas:
        changes_sim, e_sim, time_sim = _bench_sim_alpha(
            alpha,
            args.num_elements,
            rng_sim
        )
        e_analytical = cl.coding.hamming.analytical_e(alpha)
        alpha_rows.append({
            "alpha": alpha,
            "changes_sim": changes_sim,
            "e_sim": e_sim,
            "e_analytical": e_analytical,
            "time_sim": time_sim,
        })

    _print_alpha_results(alpha_rows)


if __name__ == "__main__":
    run()
