"""Show how the crystallinity parameter controls structural order.

Builds anatase and rutile supercells for several crystallinity values and
plots (a) the total g(r) of anatase and (b) the RDF similarity to the ideal
crystal together with (c) the mean Ti coordination number. Disordered
structures go through the default Matsui-Akaogi quench, so this takes a few
minutes.

    python examples/crystallinity_sweep.py            # -> docs/crystallinity_sweep.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from tio2gen import (
    build_structure,
    build_supercell,
    coordination_numbers,
    radial_distribution,
    rdf_similarity,
)

SUPERCELLS = {"anatase": (4, 4, 2), "rutile": (4, 4, 6)}
PHASE_COLORS = {"anatase": "#2a78d6", "rutile": "#eb6834"}
CHI_CURVES = [1.0, 0.75, 0.5, 0.25, 0.0]
# Ordinal blue ramp: darkest = most crystalline.
CHI_COLORS = ["#0d366b", "#1c5cab", "#2a78d6", "#5598e7", "#86b6ef"]
CHI_SWEEP = np.linspace(0.0, 1.0, 6)
SEEDS = range(2)

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def main(out=Path(__file__).resolve().parents[1] / "docs" / "crystallinity_sweep.png"):
    fig, (ax_g, ax_s, ax_cn) = plt.subplots(
        1, 3, figsize=(13, 4.2), gridspec_kw={"width_ratios": [1.3, 1, 1]}
    )
    fig.patch.set_facecolor("#fcfcfb")

    # (a) g(r) of anatase, offset vertically for readability.
    offset = 3.0
    for k, (chi, color) in enumerate(zip(CHI_CURVES, CHI_COLORS)):
        atoms = build_structure("anatase", SUPERCELLS["anatase"], chi, seed=0)
        r, g = radial_distribution(atoms, r_max=6.0)
        shift = offset * (len(CHI_CURVES) - 1 - k)
        ax_g.plot(r, g + shift, color=color, linewidth=2)
        ax_g.text(6.05, shift + 1.0, f"χ = {chi:.2f}", color=INK, va="center", fontsize=9)
    ax_g.set_xlim(1.0, 6.0)
    ax_g.set_ylim(-0.3, offset * len(CHI_CURVES) + 1.0)  # clip crystal peaks
    ax_g.set_yticks([])
    ax_g.set_xlabel("r (Å)", color=INK)
    ax_g.set_ylabel("g(r)  (curves offset)", color=INK)
    ax_g.set_title("(a) Anatase pair distribution", loc="left", color=INK)

    # (b, c) order metrics vs crystallinity, averaged over seeds.
    for phase, sc in SUPERCELLS.items():
        ref = build_supercell(phase, sc)
        sim, cn = [], []
        for chi in CHI_SWEEP:
            runs = [build_structure(phase, sc, chi, seed=s) for s in SEEDS]
            sim.append(np.mean([rdf_similarity(a, ref) for a in runs]))
            cn.append(np.mean([coordination_numbers(a).mean() for a in runs]))
        kw = dict(color=PHASE_COLORS[phase], linewidth=2, marker="o", markersize=5)
        ax_s.plot(CHI_SWEEP, sim, label=phase, **kw)
        ax_cn.plot(CHI_SWEEP, cn, label=phase, **kw)
        print(f"{phase}: similarity {sim[0]:.2f} -> {sim[-1]:.2f}, CN {cn[0]:.2f} -> {cn[-1]:.2f}")

    ax_s.set_title("(b) RDF similarity to ideal crystal", loc="left", color=INK)
    ax_s.set_ylabel("cosine similarity", color=INK)
    ax_cn.set_title("(c) Mean Ti–O coordination", loc="left", color=INK)
    ax_cn.set_ylabel("neighbours within 2.4 Å", color=INK)
    for ax in (ax_s, ax_cn):
        ax.set_xlabel("crystallinity χ", color=INK)
        ax.legend(frameon=False, labelcolor=INK)
    for ax in (ax_g, ax_s, ax_cn):
        style(ax)
        ax.set_facecolor("#fcfcfb")

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor=fig.get_facecolor())
    print(f"saved {out}")


if __name__ == "__main__":
    main()
