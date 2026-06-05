"""Benchmark Hamming LSBM embedding and simulation."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import time
import numpy as np
import conseal as cl


def _safe_div(num: float, den: float) -> float:
    if den == 0:
        return float('inf')
    return num / den


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _plot_k_results(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit(
            "matplotlib is required for plotting. Install with: pip install matplotlib"
        ) from exc

    k_vals = [row["k"] for row in rows]
    e_true = [row["e_true"] for row in rows]
    e_sim = [row["e_sim"] for row in rows]
    speedup = [
        _safe_div(row["time_true"], row["time_sim"]) if row["time_sim"] > 0 else 0.0
        for row in rows
    ]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(k_vals, e_true, marker="o", label="e_true")
    axes[0].plot(k_vals, e_sim, marker="s", label="e_sim")
    axes[0].set_xlabel("k")
    axes[0].set_ylabel("embedding efficiency (bits/change)")
    axes[0].set_title("True vs simulated efficiency")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(k_vals, speedup, marker="o", color="tab:green")
    axes[1].set_xlabel("k")
    axes[1].set_ylabel("speedup (time_true / time_sim)")
    axes[1].set_title("Simulation speedup")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _plot_alpha_results(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit(
            "matplotlib is required for plotting. Install with: pip install matplotlib"
        ) from exc

    alpha_vals = [row["alpha"] for row in rows]
    e_analytical = [row["e_analytical"] for row in rows]
    e_sim = [row["e_sim"] for row in rows]
    gain = [_safe_div(row["e_analytical"], 2.0) for row in rows]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(alpha_vals, e_analytical, marker="o", label="e_analytical")
    axes[0].plot(alpha_vals, e_sim, marker="s", label="e_sim")
    axes[0].set_xlabel("alpha")
    axes[0].set_ylabel("embedding efficiency (bits/change)")
    axes[0].set_title("Analytical vs simulated")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(alpha_vals, gain, marker="o", color="tab:purple")
    axes[1].set_xlabel("alpha")
    axes[1].set_ylabel("detectability gain (e / 2)")
    axes[1].set_title("Coding gain vs uncoded")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)

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
    parser.add_argument(
        "--csv-dir",
        default=".",
        help="Directory where CSV outputs are written."
    )
    parser.add_argument(
        "--csv-prefix",
        default="hamming_benchmark",
        help="Prefix for CSV output filenames."
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate PNG plots from benchmark results."
    )
    parser.add_argument(
        "--plot-dir",
        default=".",
        help="Directory where plot images are written."
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

    csv_dir = Path(args.csv_dir)
    _write_csv(csv_dir / f"{args.csv_prefix}_k.csv", k_rows)
    _write_csv(csv_dir / f"{args.csv_prefix}_alpha.csv", alpha_rows)
    print(
        f"\nSaved CSV: {csv_dir / f'{args.csv_prefix}_k.csv'}"
        f"\nSaved CSV: {csv_dir / f'{args.csv_prefix}_alpha.csv'}"
    )

    if args.plot:
        plot_dir = Path(args.plot_dir)
        plot_dir.mkdir(parents=True, exist_ok=True)
        plot_k_path = plot_dir / f"{args.csv_prefix}_k.png"
        plot_alpha_path = plot_dir / f"{args.csv_prefix}_alpha.png"
        _plot_k_results(k_rows, plot_k_path)
        _plot_alpha_results(alpha_rows, plot_alpha_path)
        print(
            f"\nSaved plot: {plot_k_path}"
            f"\nSaved plot: {plot_alpha_path}"
        )


if __name__ == "__main__":
    run()
