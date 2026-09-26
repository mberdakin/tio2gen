"""Crystallinity control: turning a perfect crystal into a partially amorphous one.

The model uses a single parameter, the crystallinity ``chi`` in [0, 1]:

1. A fraction ``1 - chi`` of the atoms is assigned to amorphous regions.
   These regions grow around ``n_domains`` random seeds (distances are
   computed with periodic boundary conditions), so disorder is spatially
   correlated instead of being sprinkled atom by atom.
2. Atoms in the amorphous regions receive large Gaussian displacements
   (``amorphous_sigma``); the rest may receive small thermal-like ones
   (``thermal_sigma``).
3. An iterative push-apart relaxation removes unphysical overlaps by
   enforcing species-dependent minimum distances. Crystalline atoms next
   to a disordered region may be nudged as well (interface strain).
4. Optionally (``relax_steps > 0``), a FIRE minimisation with the
   Matsui-Akaogi potential "quenches" the structure: it removes the
   artificial pile-up of distances at the hard-core values and restores
   realistic Ti-O bonding (Ti coordination ~5.5-6, d(Ti-O) ~1.93 A).

Composition and density are preserved: atoms are displaced, never added
or removed, and the cell is left untouched.
"""

from __future__ import annotations

import warnings
from typing import Dict, Optional, Tuple

import numpy as np
from ase import Atoms
from ase.geometry import get_distances
from ase.neighborlist import neighbor_list
from ase.optimize import FIRE

from .potential import MatsuiAkaogi

# Hard-core distances (Angstrom). They sit below the shortest distances of the
# crystalline phases (Ti-O 1.93, O-O 2.47, Ti-Ti 2.96) so a perfect crystal
# never triggers the relaxation.
DEFAULT_MIN_DISTANCES: Dict[Tuple[str, str], float] = {
    ("Ti", "O"): 1.70,
    ("O", "O"): 2.20,
    ("Ti", "Ti"): 2.70,
}


def _pair_table(
    atoms: Atoms, min_distances: Dict[Tuple[str, str], float]
) -> Tuple[np.ndarray, np.ndarray]:
    """Map species to indices and build a symmetric minimum-distance matrix."""
    species = sorted(set(atoms.get_chemical_symbols()))
    index = {s: k for k, s in enumerate(species)}
    table = np.zeros((len(species), len(species)))
    for (a, b), d in min_distances.items():
        if a in index and b in index:
            table[index[a], index[b]] = table[index[b], index[a]] = d
    kinds = np.array([index[s] for s in atoms.get_chemical_symbols()])
    return kinds, table


def select_amorphous_region(
    atoms: Atoms,
    fraction: float,
    n_domains: int = 1,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Boolean mask of the atoms closest to ``n_domains`` random seed points."""
    rng = np.random.default_rng(rng)
    n_amorphous = int(round(fraction * len(atoms)))
    mask = np.zeros(len(atoms), dtype=bool)
    if n_amorphous == 0:
        return mask
    seeds = rng.random((max(1, n_domains), 3)) @ atoms.cell.array
    _, dist = get_distances(
        atoms.positions, seeds, cell=atoms.cell, pbc=atoms.pbc
    )
    nearest = dist.min(axis=1)
    mask[np.argsort(nearest, kind="stable")[:n_amorphous]] = True
    return mask


def relax_overlaps(
    atoms: Atoms,
    movable: Optional[np.ndarray] = None,
    min_distances: Optional[Dict[Tuple[str, str], float]] = None,
    max_iter: int = 1000,
    tol: float = 1e-3,
) -> int:
    """Push atoms apart until every pair respects its minimum distance.

    Only atoms flagged in ``movable`` are displaced. Returns the number of
    remaining violations (0 on success). Modifies ``atoms`` in place.
    """
    min_distances = min_distances or DEFAULT_MIN_DISTANCES
    movable = (
        np.ones(len(atoms), dtype=bool) if movable is None else movable
    )
    kinds, table = _pair_table(atoms, min_distances)
    cutoff = table.max()

    n_violations = 0
    for _ in range(max_iter):
        i, j, d, vec = neighbor_list("ijdD", atoms, cutoff)
        dmin = table[kinds[i], kinds[j]]
        bad = (d < dmin - tol) & movable[i]
        n_violations = int(np.count_nonzero(bad)) // 2
        if not np.any(bad):
            return 0
        i, j, d, vec, dmin = i[bad], j[bad], d[bad], vec[bad], dmin[bad]
        # Each pair shows up as (i, j) and (j, i): split the correction
        # between both atoms when both can move.
        weight = np.where(movable[j], 0.5, 1.0)
        unit = vec / np.maximum(d, 1e-8)[:, None]
        shift = np.zeros_like(atoms.positions)
        np.add.at(shift, i, -(weight * (dmin - d))[:, None] * unit)
        atoms.positions += shift
    return n_violations


def apply_disorder(
    atoms: Atoms,
    crystallinity: float,
    n_domains: int = 1,
    amorphous_sigma: float = 0.8,
    thermal_sigma: float = 0.0,
    min_distances: Optional[Dict[Tuple[str, str], float]] = None,
    relax_steps: int = 3000,
    seed: Optional[int] = None,
) -> Atoms:
    """Return a copy of ``atoms`` with the requested degree of crystallinity.

    Parameters
    ----------
    crystallinity:
        1.0 keeps the perfect crystal, 0.0 disorders every atom.
    n_domains:
        Number of amorphous nucleation seeds. Few seeds give large amorphous
        pockets next to crystalline grains; many seeds spread the disorder.
    amorphous_sigma:
        Standard deviation (Angstrom, per Cartesian component) of the random
        displacement applied to atoms in amorphous regions.
    thermal_sigma:
        Standard deviation (Angstrom) of the displacement applied to atoms
        that remain crystalline.
    relax_steps:
        Maximum number of FIRE steps with the Matsui-Akaogi potential after
        the geometric step; the minimisation stops earlier once converged
        and warns if it runs out of steps. 0 skips it (much faster, but distances pile up at
        the hard-core values and Ti ends up under-coordinated).
    seed:
        Random seed, for reproducible structures.
    """
    if not 0.0 <= crystallinity <= 1.0:
        raise ValueError("crystallinity must be in [0, 1]")
    if n_domains < 1:
        raise ValueError("n_domains must be >= 1")

    rng = np.random.default_rng(seed)
    out = atoms.copy()
    amorphous = select_amorphous_region(
        out, 1.0 - crystallinity, n_domains=n_domains, rng=rng
    )

    sigma = np.where(amorphous, amorphous_sigma, thermal_sigma)
    out.positions += rng.normal(size=out.positions.shape) * sigma[:, None]

    if np.any(sigma > 0):
        # Crystalline atoms are allowed to move too: only those touching a
        # displaced atom will, which mimics strain at the grain boundary and
        # avoids amorphous atoms getting jammed between rigid neighbours.
        remaining = relax_overlaps(out, min_distances=min_distances)
        if remaining:
            raise RuntimeError(
                f"Could not remove {remaining} short contacts; "
                "try a smaller amorphous_sigma or looser min_distances."
            )
        if relax_steps > 0:
            quench(out, relax_steps)
    out.wrap()
    out.set_array("amorphous", amorphous.astype(int))
    out.info["crystallinity"] = float(crystallinity)
    return out


def quench(atoms: Atoms, steps: int = 3000, fmax: float = 0.05) -> int:
    """Minimise ``atoms`` in place with the Matsui-Akaogi potential.

    The cell is kept fixed. The number of steps taken and the final maximum
    force are stored in ``atoms.info`` (``quench_steps``, ``quench_fmax``),
    and a ``RuntimeWarning`` is issued if ``fmax`` is not reached within
    ``steps``. Returns the number of FIRE steps taken.
    """
    atoms.calc = MatsuiAkaogi()
    opt = FIRE(atoms, logfile=None, maxstep=0.1)
    opt.run(fmax=fmax, steps=steps)
    final_fmax = float(np.linalg.norm(atoms.get_forces(), axis=1).max())
    atoms.calc = None
    atoms.info["quench_steps"] = opt.nsteps
    atoms.info["quench_fmax"] = round(final_fmax, 4)
    if final_fmax > fmax:
        warnings.warn(
            f"Quench did not converge in {opt.nsteps} steps (max force "
            f"{final_fmax:.3f} eV/A > {fmax} eV/A); increase relax_steps.",
            RuntimeWarning,
            stacklevel=2,
        )
    return opt.nsteps
