"""Structural descriptors used to check how ordered a structure really is."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from ase import Atoms
from ase.neighborlist import neighbor_list


def radial_distribution(
    atoms: Atoms,
    r_max: float = 6.0,
    n_bins: int = 240,
    pair: Optional[Tuple[str, str]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Radial distribution function g(r) under periodic boundary conditions.

    ``pair=("Ti", "O")`` restricts the histogram to a partial g(r).
    ``r_max`` must be smaller than half the shortest cell height.
    """
    i, j, d = neighbor_list("ijd", atoms, r_max)
    symbols = np.array(atoms.get_chemical_symbols())
    if pair is None:
        n_i = n_j = len(atoms)
    else:
        keep = (symbols[i] == pair[0]) & (symbols[j] == pair[1])
        d = d[keep]
        n_i = np.count_nonzero(symbols == pair[0])
        n_j = np.count_nonzero(symbols == pair[1])

    edges = np.linspace(0.0, r_max, n_bins + 1)
    counts, _ = np.histogram(d, bins=edges)
    r = 0.5 * (edges[1:] + edges[:-1])
    shell = 4.0 / 3.0 * np.pi * (edges[1:] ** 3 - edges[:-1] ** 3)
    density = n_j / atoms.get_volume()
    return r, counts / (n_i * density * shell)


def coordination_numbers(atoms: Atoms, cutoff: float = 2.4) -> np.ndarray:
    """Number of O neighbours within ``cutoff`` of every Ti atom."""
    i, j = neighbor_list("ij", atoms, cutoff)
    symbols = np.array(atoms.get_chemical_symbols())
    ti = np.flatnonzero(symbols == "Ti")
    bonds = i[(symbols[i] == "Ti") & (symbols[j] == "O")]
    return np.bincount(bonds, minlength=len(atoms))[ti]


def rdf_similarity(
    atoms: Atoms, reference: Atoms, r_max: float = 6.0, smear: float = 0.1
) -> float:
    """Cosine similarity between the Gaussian-smeared g(r) of two structures.

    1.0 means the same pair distribution as ``reference`` (e.g. the ideal
    crystal); values drop as long-range order is lost.
    """
    r, g = radial_distribution(atoms, r_max)
    _, g_ref = radial_distribution(reference, r_max)
    dr = r[1] - r[0]
    x = np.arange(-4 * smear, 4 * smear + dr, dr)
    kernel = np.exp(-0.5 * (x / smear) ** 2)
    g = np.convolve(g - 1.0, kernel, mode="same")
    g_ref = np.convolve(g_ref - 1.0, kernel, mode="same")
    return float(g @ g_ref / (np.linalg.norm(g) * np.linalg.norm(g_ref)))
