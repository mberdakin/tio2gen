"""Crystallographic data and unit-cell builders for TiO2 polymorphs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

from ase import Atoms
from ase.spacegroup import crystal


@dataclass(frozen=True)
class PhaseData:
    """Minimal crystallographic description of a TiO2 polymorph."""

    name: str
    spacegroup: int
    setting: int
    a: float
    c: float
    ti_site: Tuple[float, float, float]
    o_site: Tuple[float, float, float]
    reference: str


# Experimental lattice parameters (room temperature).
PHASES: Dict[str, PhaseData] = {
    "anatase": PhaseData(
        name="anatase",
        spacegroup=141,  # I4_1/amd
        setting=1,
        a=3.7845,
        c=9.5143,
        ti_site=(0.0, 0.0, 0.0),
        o_site=(0.0, 0.0, 0.2081),
        reference="Burdett et al., J. Am. Chem. Soc. 109, 3639 (1987)",
    ),
    "rutile": PhaseData(
        name="rutile",
        spacegroup=136,  # P4_2/mnm
        setting=1,
        a=4.5937,
        c=2.9587,
        ti_site=(0.0, 0.0, 0.0),
        o_site=(0.3053, 0.3053, 0.0),
        reference="Burdett et al., J. Am. Chem. Soc. 109, 3639 (1987)",
    ),
}


def available_phases() -> Tuple[str, ...]:
    return tuple(PHASES)


def unit_cell(phase: str) -> Atoms:
    """Return the conventional unit cell of the requested TiO2 phase.

    Anatase has 12 atoms (Ti4O8) and rutile 6 atoms (Ti2O4) per cell.
    """
    key = phase.strip().lower()
    if key not in PHASES:
        raise ValueError(
            f"Unknown phase '{phase}'. Choose one of: {', '.join(PHASES)}"
        )
    data = PHASES[key]
    atoms = crystal(
        symbols=["Ti", "O"],
        basis=[data.ti_site, data.o_site],
        spacegroup=data.spacegroup,
        setting=data.setting,
        cellpar=[data.a, data.a, data.c, 90, 90, 90],
        primitive_cell=False,
    )
    # Symmetry metadata stops being true as soon as disorder is applied.
    atoms.info = {"phase": data.name}
    del atoms.arrays["spacegroup_kinds"]
    return atoms
