"""Benchmark Hamming LSBM embedding and simulation."""
from __future__ import annotations
"""@copyright: large parts of the benchmarking script where written with the help of gemini 3.1. Pro"""

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


def _plot_k_results(rows: list[dict], plot_dir: Path, prefix: str) -> None:
    if not rows:
        return
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        print("matplotlib is required for plotting. Install with: pip install matplotlib")
        return

    k_vals = [row["k"] for row in rows]
    e_true = [row["e_true"] for row in rows]
    e_sim = [row["e_sim"] for row in rows]
    speedup = [
        _safe_div(row["time_true"], row["time_sim"]) if row["time_sim"] > 0 else 0.0
        for row in rows
    ]

    # Plot 1: Efficiency
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(k_vals, e_true, marker="o", label="e_true")
    ax1.plot(k_vals, e_sim, marker="s", label="e_sim")
    ax1.set_xlabel("k")
    ax1.set_ylabel("Embedding Efficiency (bits/change)")
    ax1.set_title("True vs Simulated Efficiency")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    fig1.tight_layout()
    fig1.savefig(plot_dir / f"{prefix}_k_efficiency.png", dpi=200)
    plt.close(fig1)

    # Plot 2: Speedup
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.plot(k_vals, speedup, marker="o", color="tab:green")
    ax2.set_xlabel("k")
    ax2.set_ylabel("Speedup (time_true / time_sim)")
    ax2.set_title("Simulation Speedup")
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(plot_dir / f"{prefix}_k_speedup.png", dpi=200)
    plt.close(fig2)


def _plot_hcf_com_sim_results(rows: list[dict], plot_dir: Path, prefix: str) -> None:
    if not rows:
        return
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    alpha_vals = [row["alpha"] for row in rows]
    pe_coded = [row["pe_coded"] for row in rows]
    pe_uncoded = [row["pe_uncoded"] for row in rows]
    beta_coded = [row["beta_coded"] for row in rows]
    beta_uncoded = [row["beta_uncoded"] for row in rows]

    # Plot 1: Probability of Error
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(alpha_vals, pe_coded, marker="o", color="tab:blue", label="P_E (Coded)")
    ax1.plot(alpha_vals, pe_uncoded, marker="x", color="tab:red", label="P_E (Uncoded)")
    ax1.set_xlabel("Alpha (bits/element)")
    ax1.set_ylabel("Probability of Error (P_E)")
    ax1.set_title("Simulated HCF-COM Attack Performance")
    ax1.axhline(y=0.5, color="k", linestyle="--", alpha=0.5)
    ax1.set_ylim([0, 0.55])
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    fig1.tight_layout()
    fig1.savefig(plot_dir / f"{prefix}_hcf_sim_pe.png", dpi=200)
    plt.close(fig1)

    # Plot 2: Change Rates
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.plot(alpha_vals, beta_coded, marker="s", color="tab:blue", label="Coded (Analytical)")
    ax2.plot(alpha_vals, beta_uncoded, marker="^", color="tab:red", label="Uncoded (e=2)")
    ax2.set_xlabel("Alpha (bits/element)")
    ax2.set_ylabel("Change Rate (Beta)")
    ax2.set_title("Simulated Embedding Changes")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig(plot_dir / f"{prefix}_hcf_sim_beta.png", dpi=200)
    plt.close(fig2)


def _plot_hcf_com_real_results(rows: list[dict], plot_dir: Path, prefix: str) -> None:
    if not rows:
        return
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    alpha_vals = [row["alpha"] for row in rows]
    pe_coded = [row["pe_coded"] for row in rows]
    pe_uncoded = [row["pe_uncoded"] for row in rows]

    fig1, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(alpha_vals, pe_coded, marker="o", color="tab:blue", label="P_E (Real Coded)")
    ax1.plot(alpha_vals, pe_uncoded, marker="x", color="tab:red", label="P_E (Real Uncoded)")
    ax1.set_xlabel("Alpha (bits/element)")
    ax1.set_ylabel("Probability of Error (P_E)")
    ax1.set_title("Real Image HCF-COM Attack Performance")
    ax1.axhline(y=0.5, color="k", linestyle="--", alpha=0.5)
    ax1.set_ylim([0, 0.55])
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    fig1.tight_layout()
    fig1.savefig(plot_dir / f"{prefix}_hcf_real_pe.png", dpi=200)
    plt.close(fig1)


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


def _hcf_com(x: np.ndarray) -> float:
    """Calculate the Histogram Characteristic Function Center of Mass."""
    h, _ = np.histogram(x, bins=256, range=(0, 256))
    H = np.abs(np.fft.fft(h))
    H_half = H[:128]
    denom = np.sum(H_half)
    if denom == 0:
        return 0.0
    com = np.sum(np.arange(128) * H_half) / denom
    return float(com)


def _calc_pe(features_cover: np.ndarray, features_stego: np.ndarray) -> float:
    """Calculate the minimum probability of error P_E by finding the optimal threshold."""
    features = np.concatenate([features_cover, features_stego])
    labels = np.concatenate([np.zeros(len(features_cover)), np.ones(len(features_stego))])

    sort_idx = np.argsort(features)
    sorted_labels = labels[sort_idx]

    n_cover = len(features_cover)
    n_stego = len(features_stego)

    min_pe = 1.0

    fa_count = 0  
    md_count = n_stego  

    for label in sorted_labels:
        pe = (fa_count / n_cover + md_count / n_stego) / 2.0
        if pe < min_pe:
            min_pe = pe
        if label == 0:
            fa_count += 1
        else:
            md_count -= 1
            
    pe = (fa_count / n_cover + md_count / n_stego) / 2.0
    if pe < min_pe:
        min_pe = pe

    fa_count = n_cover
    md_count = 0
    min_pe_rev = 1.0

    for label in sorted_labels:
        pe = (fa_count / n_cover + md_count / n_stego) / 2.0
        if pe < min_pe_rev:
            min_pe_rev = pe
        if label == 0:
            fa_count -= 1
        else:
            md_count += 1

    pe = (fa_count / n_cover + md_count / n_stego) / 2.0
    if pe < min_pe_rev:
        min_pe_rev = pe

    return float(min(min_pe, min_pe_rev))


def _bench_hcf_com_simulated(
    alphas: list[float],
    num_images: int,
    image_size: int,
    rng: np.random.Generator
) -> list[dict]:
    """Perform HCF-COM attack on SIMULATED sequences."""
    print(f"\nGenerating {num_images} simulated covers (size {image_size}) for HCF-COM attack...")
    covers = []
    for _ in range(num_images):
        noise = rng.normal(0, 5, size=image_size)
        walk = np.cumsum(noise)
        walk = walk - np.min(walk)
        if np.max(walk) > 0:
            walk = (walk / np.max(walk)) * 255
        covers.append(walk.astype(np.uint8))
    
    features_cover = np.array([_hcf_com(c) for c in covers])
    
    rows = []
    for alpha in alphas:
        start = time.perf_counter()
        
        try:
            e_analytical = cl.coding.hamming.analytical_e(alpha)
            beta_coded = alpha / e_analytical
        except Exception:
            beta_coded = alpha / 2.0  
            
        beta_uncoded = alpha / 2.0
        
        features_stego_coded = []
        features_stego_uncoded = []
        
        for cover in covers:
            # Simulated Coded
            stego_c = cover.copy().astype(np.int16)
            change_mask_c = rng.random(size=stego_c.shape) < beta_coded
            step_c = rng.choice([-1, 1], size=stego_c.shape)
            stego_c[change_mask_c] += step_c[change_mask_c]
            features_stego_coded.append(_hcf_com(np.clip(stego_c, 0, 255).astype(np.uint8)))
            
            # Simulated Uncoded
            stego_u = cover.copy().astype(np.int16)
            change_mask_u = rng.random(size=stego_u.shape) < beta_uncoded
            step_u = rng.choice([-1, 1], size=stego_u.shape)
            stego_u[change_mask_u] += step_u[change_mask_u]
            features_stego_uncoded.append(_hcf_com(np.clip(stego_u, 0, 255).astype(np.uint8)))
        
        pe_coded = _calc_pe(features_cover, np.array(features_stego_coded))
        pe_uncoded = _calc_pe(features_cover, np.array(features_stego_uncoded))
        elapsed = time.perf_counter() - start
        
        rows.append({
            "alpha": alpha,
            "beta_coded": beta_coded,
            "beta_uncoded": beta_uncoded,
            "pe_coded": pe_coded,
            "pe_uncoded": pe_uncoded,
            "time_attack": elapsed
        })
    return rows


def _true_embed_and_verify(cover: np.ndarray, alpha: float, rng: np.random.Generator) -> np.ndarray:
    """Perform true embedding, interpolating codes for arbitrary alpha, and verify extraction."""
    stego = cover.copy()
    try:
        k, k1, weight_k, _ = cl.coding.hamming._mix_for_alpha(alpha)
        n_k = 2**k - 1
        n_k1 = 2**k1 - 1
    except Exception:
        return _true_embed_uncoded_and_verify(cover, alpha, rng)

    split_idx = int(len(cover) * weight_k)

    # Part 1: Embed with code k
    cover_k = stego[:split_idx]
    blocks_k = len(cover_k) // n_k
    if blocks_k > 0:
        msg_bytes_k = max(1, (blocks_k * k) // 8)
        msg_k = rng.integers(0, 256, size=msg_bytes_k, dtype=np.uint8)
        msg_bits_k = np.unpackbits(msg_k)
        
        stego_k = cl.coding.hamming.embed_message_lsbm_hamming(cover_k, msg_k, k, rng)
        stego[:split_idx] = stego_k
        
        # Verify
        extracted_bits_k = cl.coding.hamming.extract_lsbm_hamming(stego_k, k, num_bits=len(msg_bits_k))
        assert np.array_equal(msg_bits_k, extracted_bits_k), f"Extraction failed for k={k}"

    # Part 2: Embed with code k1
    cover_k1 = stego[split_idx:]
    blocks_k1 = len(cover_k1) // n_k1
    if blocks_k1 > 0:
        msg_bytes_k1 = max(1, (blocks_k1 * k1) // 8)
        msg_k1 = rng.integers(0, 256, size=msg_bytes_k1, dtype=np.uint8)
        msg_bits_k1 = np.unpackbits(msg_k1)
        
        stego_k1 = cl.coding.hamming.embed_message_lsbm_hamming(cover_k1, msg_k1, k1, rng)
        stego[split_idx:] = stego_k1
        
        # Verify
        extracted_bits_k1 = cl.coding.hamming.extract_lsbm_hamming(stego_k1, k1, num_bits=len(msg_bits_k1))
        assert np.array_equal(msg_bits_k1, extracted_bits_k1), f"Extraction failed for k={k1}"

    return stego


def _true_embed_uncoded_and_verify(cover: np.ndarray, alpha: float, rng: np.random.Generator) -> np.ndarray:
    """Perform true uncoded LSBM and verify extraction."""
    stego = cover.copy().astype(np.int16)
    n_embed = int(len(stego) * alpha)
    
    if n_embed == 0:
        return cover.copy()
        
    msg_bits = rng.integers(0, 2, size=n_embed, dtype=np.uint8)
    current_lsbs = stego[:n_embed] % 2
    diff = (msg_bits - current_lsbs) % 2
    
    change_mask = diff == 1
    
    for i in np.where(change_mask)[0]:
        val = stego[i]
        if val <= 0:
            stego[i] = 1
        elif val >= 255:
            stego[i] = 254
        else:
            stego[i] += rng.choice([-1, 1])
            
    stego_uint8 = np.clip(stego, 0, 255).astype(np.uint8)
    
    # Verify
    extracted_lsbs = stego_uint8[:n_embed] % 2
    assert np.array_equal(msg_bits, extracted_lsbs), f"Uncoded extraction failed for alpha={alpha}"
    
    return stego_uint8


def _bench_hcf_com_real(
    alphas: list[float],
    image_dir: Path,
    max_images: int,
    rng: np.random.Generator
) -> list[dict]:
    """Perform HCF-COM attack on REAL images using TRUE embedding & extraction verification."""
    try:
        from PIL import Image
    except ImportError:
        print("PIL is not installed. Skipping real image benchmark.")
        return []

    covers = []
    if image_dir.exists() and image_dir.is_dir():
        valid_exts = {'.pgm', '.png', '.jpg', '.jpeg'}
        files = [p for p in image_dir.iterdir() if p.suffix.lower() in valid_exts]
        files = sorted(files)[:max_images]
        for f in files:
            try:
                img = np.array(Image.open(f).convert('L')).flatten()
                covers.append(img)
            except Exception as e:
                print(f"Failed to load {f}: {e}")

    if not covers:
        print(f"\nWarning: No valid images found in {image_dir}. Skipping real image benchmark.")
        return []

    print(f"\nLoaded {len(covers)} real images from {image_dir} for TRUE HCF-COM attack...")
    features_cover = np.array([_hcf_com(c) for c in covers])
    
    rows = []
    for alpha in alphas:
        start = time.perf_counter()
        features_stego_coded = []
        features_stego_uncoded = []
        
        for cover in covers:
            # True Coded (with extraction verification embedded in the function)
            stego_c = _true_embed_and_verify(cover, alpha, rng)
            features_stego_coded.append(_hcf_com(stego_c))
            
            # True Uncoded
            stego_u = _true_embed_uncoded_and_verify(cover, alpha, rng)
            features_stego_uncoded.append(_hcf_com(stego_u))
        
        pe_coded = _calc_pe(features_cover, np.array(features_stego_coded))
        pe_uncoded = _calc_pe(features_cover, np.array(features_stego_uncoded))
        elapsed = time.perf_counter() - start
        
        rows.append({
            "alpha": alpha,
            "pe_coded": pe_coded,
            "pe_uncoded": pe_uncoded,
            "time_attack": elapsed
        })
    return rows


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


def _print_hcf_com_results(title: str, rows: list[dict]) -> None:
    if not rows:
        return
    print(f"\n{title}")
    print("alpha   P_E(Coded)  P_E(Uncoded)  t_attack_s")
    for row in rows:
        print(
            f"{row['alpha']:.4f}  {row['pe_coded']:.4f}      "
            f"{row['pe_uncoded']:.4f}        {row['time_attack']:>8.3f}"
        )


def run() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark Hamming LSBM embedding and HCF-COM attacks."
    )
    parser.add_argument("--k", type=int, nargs="*", default=[2, 3, 4, 5, 6])
    parser.add_argument("--num-blocks", type=int, default=1000)
    parser.add_argument("--alphas", type=float, nargs="*", default=[0.1, 0.2, 0.4, 0.6, 0.8])
    parser.add_argument("--num-images-sim", type=int, default=500, help="Number of simulated covers.")
    parser.add_argument("--num-elements-sim", type=int, default=100000, help="Size of simulated covers.")
    parser.add_argument("--bossbase-dir", default="./BOSSbase_1.01", help="Directory with real images.")
    parser.add_argument("--max-real-images", type=int, default=500, help="Max real images to test.")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--csv-dir", default=".")
    parser.add_argument("--csv-prefix", default="hamming_benchmark")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--plot-dir", default=".")
    args = parser.parse_args()

    seed_seq = np.random.SeedSequence(args.seed)
    rngs = [np.random.default_rng(s) for s in seed_seq.spawn(4)]
    rng_cover, rng_msg, rng_embed, rng_sim = rngs

    # Phase 1: k-based efficiency benchmark
    k_rows = []
    for k in args.k:
        n, changes_true, e_true, time_true = _bench_true(k, args.num_blocks, rng_cover, rng_msg, rng_embed)
        changes_sim, e_sim, time_sim = _bench_sim_k(k, args.num_blocks, rng_sim)
        alpha = k / n
        k_rows.append({
            "k": k, "n": n, "alpha": alpha,
            "changes_true": changes_true, "changes_sim": changes_sim,
            "e_true": e_true, "e_sim": e_sim,
            "time_true": time_true, "time_sim": time_sim,
        })

    _print_k_results(k_rows)

    # Phase 2: Simulated Sequences HCF-COM (beta simulation)
    hcf_sim_rows = _bench_hcf_com_simulated(args.alphas, args.num_images_sim, args.num_elements_sim, rng_sim)
    _print_hcf_com_results("HCF-COM Attack Probability of Error (P_E) - SIMULATED", hcf_sim_rows)

    # Phase 3: Real Image HCF-COM (true embedding & extraction verification)
    hcf_real_rows = _bench_hcf_com_real(args.alphas, Path(args.bossbase_dir), args.max_real_images, rng_sim)
    _print_hcf_com_results("HCF-COM Attack Probability of Error (P_E) - REAL IMAGES (Verified)", hcf_real_rows)

    csv_dir = Path(args.csv_dir)
    _write_csv(csv_dir / f"{args.csv_prefix}_k.csv", k_rows)
    _write_csv(csv_dir / f"{args.csv_prefix}_hcf_com_sim.csv", hcf_sim_rows)
    if hcf_real_rows:
        _write_csv(csv_dir / f"{args.csv_prefix}_hcf_com_real.csv", hcf_real_rows)

    print(f"\nSaved CSV: {csv_dir / f'{args.csv_prefix}_k.csv'}")
    print(f"Saved CSV: {csv_dir / f'{args.csv_prefix}_hcf_com_sim.csv'}")
    if hcf_real_rows:
        print(f"Saved CSV: {csv_dir / f'{args.csv_prefix}_hcf_com_real.csv'}")

    # Plotting Phase
    if args.plot:
        plot_dir = Path(args.plot_dir)
        plot_dir.mkdir(parents=True, exist_ok=True)
        _plot_k_results(k_rows, plot_dir, args.csv_prefix)
        _plot_hcf_com_sim_results(hcf_sim_rows, plot_dir, args.csv_prefix)
        if hcf_real_rows:
            _plot_hcf_com_real_results(hcf_real_rows, plot_dir, args.csv_prefix)
        print("\nSaved individual plots to:", plot_dir.resolve())

if __name__ == "__main__":
    run()