"""Quick structural check of one or more TiO2 structures.

For every file (any format ASE reads: XYZ, GEN, CIF, DFTB+ geo_end.gen, ...)
prints the mean Ti coordination, the g(r) similarity to the ideal crystal and
the coordination histogram, and plots all the g(r) curves together.

    python examples/analyze_structures.py structures/*.gen
    python examples/analyze_structures.py -p rutile -s 4x4x6 -o rutile_gr.png *.xyz

The reference crystal should match the structures being analysed (same phase
and supercell), otherwise the similarity is meaningless.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from ase.io import read

from tio2gen import (
    available_phases,
    build_supercell,
    coordination_numbers,
    radial_distribution,
    rdf_similarity,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("files", nargs="+", help="structure files to analyse")
    parser.add_argument(
        "-p", "--phase", choices=available_phases(), default="anatase",
        help="phase of the reference crystal (default: anatase)",
    )
    parser.add_argument(
        "-s", "--supercell", default="6x6x3", metavar="NxNxN",
        help="supercell of the reference crystal, e.g. 6x6x3 or 4 (default: 6x6x3)",
    )
    parser.add_argument(
        "-o", "--output", default="gr.png", help="g(r) plot file (default: gr.png)"
    )
    args = parser.parse_args()

    reps = [int(n) for n in args.supercell.lower().split("x")]
    ref = build_supercell(args.phase, reps[0] if len(reps) == 1 else reps)

    for path in args.files:
        atoms = read(path)
        cn = coordination_numbers(atoms)
        histogram = {int(k): int(n) for k, n in zip(*np.unique(cn, return_counts=True))}
        print(
            f"{path}: CN = {cn.mean():.2f}, "
            f"similarity = {rdf_similarity(atoms, ref):.3f}, "
            f"CN histogram = {histogram}"
        )
        r, g = radial_distribution(atoms, r_max=8.0)
        plt.plot(r, g, label=Path(path).name)

    plt.xlabel("r (Å)")
    plt.ylabel("g(r)")
    plt.legend()
    plt.savefig(args.output, dpi=150)
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
