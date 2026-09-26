"""Command-line interface: ``tio2gen --phase rutile --supercell 3 3 4 -c 0.5``."""

from __future__ import annotations

import argparse
from typing import List, Optional

from . import __version__
from .builder import build_structure
from .io import FORMATS, write_structure
from .phases import available_phases


def _crystallinity(value: str) -> float:
    chi = float(value)
    if not 0.0 <= chi <= 1.0:
        raise argparse.ArgumentTypeError("crystallinity must be in [0, 1]")
    return chi


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tio2gen",
        description="Generate TiO2 (anatase/rutile) supercells with tunable "
        "crystallinity and write them as XYZ, CIF and DFTB+ GEN files.",
    )
    parser.add_argument(
        "-p", "--phase", choices=available_phases(), default="anatase",
        help="TiO2 polymorph (default: anatase)",
    )
    parser.add_argument(
        "-s", "--supercell", type=int, nargs="+", default=[1], metavar="N",
        help="repetitions of the unit cell: one value (N N N) or three "
        "(NX NY NZ). Default: 1",
    )
    parser.add_argument(
        "-c", "--crystallinity", type=_crystallinity, nargs="+", default=[1.0],
        metavar="CHI",
        help="crystallinity in [0, 1]; 1 = perfect crystal, 0 = fully "
        "disordered. Several values produce one structure each.",
    )
    parser.add_argument(
        "--domains", type=int, default=1,
        help="number of amorphous nucleation seeds (default: 1)",
    )
    parser.add_argument(
        "--amorphous-sigma", type=float, default=0.8, metavar="ANG",
        help="displacement amplitude in amorphous regions, in Angstrom "
        "(default: 0.8)",
    )
    parser.add_argument(
        "--thermal-sigma", type=float, default=0.0, metavar="ANG",
        help="displacement amplitude in crystalline regions, in Angstrom "
        "(default: 0)",
    )
    parser.add_argument(
        "--relax-steps", type=int, default=3000, metavar="N",
        help="max FIRE steps of the Matsui-Akaogi quench applied to disordered "
        "structures; it stops once converged. 0 disables it (default: 3000)",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="random seed for reproducibility"
    )
    parser.add_argument(
        "-f", "--formats", nargs="+", choices=list(FORMATS),
        default=list(FORMATS), help="output formats (default: all)",
    )
    parser.add_argument(
        "-o", "--outdir", default="structures",
        help="output directory (default: ./structures)",
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if len(args.supercell) == 1:
        supercell = args.supercell * 3
    elif len(args.supercell) == 3:
        supercell = args.supercell
    else:
        parser.error("--supercell takes one or three integers")

    for chi in args.crystallinity:
        atoms = build_structure(
            phase=args.phase,
            supercell=supercell,
            crystallinity=chi,
            n_domains=args.domains,
            amorphous_sigma=args.amorphous_sigma,
            thermal_sigma=args.thermal_sigma,
            relax_steps=args.relax_steps,
            seed=args.seed,
        )
        paths = write_structure(atoms, args.outdir, args.formats)
        print(
            f"{atoms.get_chemical_formula(mode='metal')} ({len(atoms)} atoms, "
            f"chi={chi:.2f}) -> {', '.join(str(p) for p in paths)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
