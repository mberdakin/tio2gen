"""High-level API: phase + supercell + crystallinity -> ASE Atoms."""

from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np
from ase import Atoms

from .disorder import apply_disorder
from .phases import unit_cell

SupercellSpec = Union[int, Sequence[int]]


def _parse_supercell(supercell: SupercellSpec):
    if isinstance(supercell, int):
        reps = (supercell,) * 3
    else:
        reps = tuple(int(n) for n in supercell)
    if len(reps) != 3 or any(n < 1 for n in reps):
        raise ValueError(
            "supercell must be a positive integer or three positive integers"
        )
    return reps


def build_supercell(phase: str, supercell: SupercellSpec = 1) -> Atoms:
    """Perfect crystalline supercell of ``phase`` repeated ``supercell`` times."""
    reps = _parse_supercell(supercell)
    atoms = unit_cell(phase).repeat(reps)
    # Group atoms by species (Ti block, then O block).
    atoms = atoms[np.argsort(-atoms.numbers, kind="stable")]
    atoms.info["supercell"] = "x".join(str(n) for n in reps)
    return atoms


def build_structure(
    phase: str = "anatase",
    supercell: SupercellSpec = 1,
    crystallinity: float = 1.0,
    n_domains: int = 1,
    amorphous_sigma: float = 0.8,
    thermal_sigma: float = 0.0,
    relax_steps: int = 3000,
    seed: Optional[int] = None,
) -> Atoms:
    """Build a TiO2 supercell with a tunable degree of crystallinity.

    Parameters
    ----------
    phase:
        ``"anatase"`` or ``"rutile"``.
    supercell:
        Repetitions of the conventional cell, either ``n`` or ``(nx, ny, nz)``.
    crystallinity:
        1.0 gives the perfect crystal; lower values progressively turn the
        structure amorphous (see :func:`tio2gen.disorder.apply_disorder`).
    """
    atoms = build_supercell(phase, supercell)
    if crystallinity == 1.0 and thermal_sigma == 0.0:
        atoms.set_array("amorphous", np.zeros(len(atoms), dtype=int))
        atoms.info["crystallinity"] = 1.0
    else:
        atoms = apply_disorder(
            atoms,
            crystallinity,
            n_domains=n_domains,
            amorphous_sigma=amorphous_sigma,
            thermal_sigma=thermal_sigma,
            relax_steps=relax_steps,
            seed=seed,
        )
    if seed is not None:
        atoms.info["seed"] = seed
    return atoms
