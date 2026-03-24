"""
Extract last-layer Transformer attention weights per head for a formula.

Usage:
    python inference/extract_attention.py Li6PS5Cl
    python inference/extract_attention.py Fe2O3 -o attention.npy
    python inference/extract_attention.py "Na2Ga11O17" --output-dir outputs

Saves one heatmap PNG per head (e.g. Li6PS5Cl_head1.png) when matplotlib is available.
"""
import argparse
import json
import os
import re
import sys

import numpy as np

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from inference.predictor import FormationEnergyPredictor

PLOT_FONT_SIZE = 25


def main():
    parser = argparse.ArgumentParser(
        description="Extract last-layer attention weights per head; save optional .npy/.npz and heatmaps."
    )
    parser.add_argument("formula", type=str, help="Chemical formula, e.g. Li6PS5Cl, Fe2O3")
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output path (.npy or .npz; .npz stores elements + formula)",
    )
    parser.add_argument("--output-dir", type=str, default=".", help="Directory for heatmap PNGs")
    parser.add_argument("--model-path", type=str, default="checkpoints/best_model_final.pt",
                        help="Model checkpoint path")
    args = parser.parse_args()

    formula = args.formula.strip()
    if not formula:
        print("Error: formula must not be empty")
        sys.exit(1)

    print(f"Loading model: {args.model_path}")
    predictor = FormationEnergyPredictor(model_path=args.model_path)
    print(f"Parsing formula: {formula}")

    attn_per_head, elements = predictor.get_attention_weights_per_head(formula)
    n_heads = attn_per_head.shape[0]

    print(f"\nElement order: {elements}")
    print(f"Attention shape: [{n_heads} heads, {attn_per_head.shape[1]}, {attn_per_head.shape[2]}]")

    if args.output:
        out_path = args.output
        if out_path.endswith(".npz"):
            np.savez(
                out_path,
                attention_per_head=attn_per_head,
                elements=np.array(elements, dtype=object),
                formula=formula,
            )
            print(f"\nSaved: {out_path} (attention_per_head, elements, formula)")
        else:
            np.save(out_path, attn_per_head)
            meta_path = out_path.replace(".npy", "_elements.json")
            if meta_path == out_path:
                meta_path = out_path + "_elements.json"
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump({"formula": formula, "elements": elements}, f, indent=2)
            print(f"\nSaved: {out_path} (array), {meta_path} (element order)")

    formula_safe = re.sub(r"[^\w\-]", "_", formula.strip()) or "formula"
    os.makedirs(args.output_dir, exist_ok=True)

    try:
        import matplotlib.pyplot as plt

        fs = PLOT_FONT_SIZE
        plt.rcParams.update({
            "font.size": fs,
            "axes.titlesize": fs,
            "axes.labelsize": fs,
            "xtick.labelsize": fs,
            "ytick.labelsize": fs,
        })

        saved_paths = []
        for h in range(n_heads):
            head_num = h + 1
            attn_h = attn_per_head[h]
            plot_filename = f"{formula_safe}_head{head_num}.png"
            plot_path = os.path.join(args.output_dir, plot_filename)

            fig, ax = plt.subplots(figsize=(6, 5))
            im = ax.imshow(attn_h, cmap="Blues", aspect="auto", vmin=0, vmax=1)
            ax.set_xticks(range(len(elements)))
            ax.set_yticks(range(len(elements)))
            ax.set_xticklabels(elements)
            ax.set_yticklabels(elements)
            ax.set_xlabel("Key")
            ax.set_ylabel("Query")
            ax.set_title(f"Head {head_num}")
            cbar = plt.colorbar(im, ax=ax)
            cbar.ax.tick_params(labelsize=fs)
            plt.tight_layout()
            plt.savefig(plot_path, dpi=150, bbox_inches="tight")
            plt.close()
            saved_paths.append(plot_path)

        print(f"\nSaved {n_heads} head heatmaps:")
        for p in saved_paths:
            print(f"  - {p}")
    except ImportError:
        print("\nPlotting skipped: matplotlib not installed")


if __name__ == "__main__":
    main()
