"""
Read results/20_ad_miga.json and print the filled LaTeX table rows
for the AD-MIGA section in 4_results_short.tex.
"""

import json
from pathlib import Path

ROOT = Path(__file__).parent.parent
IN_FILE = ROOT / "results" / "20_ad_miga.json"

DATASETS = ["Iris", "Haberman", "Wholesale"]
VARIANTS = [("STD", "MIGA"), ("L", "AD-MIGA-L"), ("E", "AD-MIGA-E"), ("MICE", "MICE")]

def main():
    with open(IN_FILE) as f:
        data = json.load(f)

    print("% ── Filled table rows for tab:ad_miga ─────────────────────────────")
    for ds in DATASETS:
        print(f"\\multirow{{4}}{{*}}{{{ds}}}")
        for key, label in VARIANTS:
            r = data[ds][key]
            fr_m = r["fr_mean"]
            fr_s = r["fr_std"]
            rm_m = r["rmse_mean"]
            cg = f"{r['conv_gen_mean']:.0f}" if r["conv_gen_mean"] is not None else "---"
            print(f" & {label:<12s} & {fr_m:.4f}~$\\pm$~{fr_s:.4f} & {rm_m:.4f} & {cg} \\\\")
        if ds != DATASETS[-1]:
            print("\\midrule")

    print()
    print("% ── Key findings for figure captions ───────────────────────────────")
    # Find how many datasets AD-MIGA-E beats STD on Fr
    better_on = []
    all_conv_gains = []
    for ds in DATASETS:
        std_fr = data[ds]["STD"]["fr_mean"]
        e_fr   = data[ds]["E"]["fr_mean"]
        if e_fr < std_fr:
            better_on.append(ds)
        cg_std = data[ds]["STD"]["conv_gen_mean"]
        cg_e   = data[ds]["E"]["conv_gen_mean"]
        if cg_std is not None and cg_e is not None:
            all_conv_gains.append(cg_std - cg_e)

    print(f"AD-MIGA-E better on Fr: {len(better_on)}/{len(DATASETS)} datasets ({', '.join(better_on)})")
    if all_conv_gains:
        avg_gain = sum(all_conv_gains) / len(all_conv_gains)
        print(f"Average convergence speedup (STD - E conv_gen): {avg_gain:.1f} generations")

if __name__ == "__main__":
    main()
