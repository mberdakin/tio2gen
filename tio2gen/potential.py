"""Matsui-Akaogi pair potential for TiO2 as an ASE calculator.

    U(r) = q_i q_j / r + A_ij exp(-r / rho_ij) - C_ij / r^6

Coulomb interactions use the damped shifted force (DSF) scheme of Fennell &
Gezelter (J. Chem. Phys. 124, 234104, 2006), a real-space alternative to
Ewald summation whose energy and force both vanish smoothly at the cutoff.
It is meant for quick geometry relaxations of disordered TiO2, not as a
replacement for DFTB+.

Reference: M. Matsui and M. Akaogi, Mol. Simul. 6, 239 (1991).
"""

from __future__ import annotations

from math import erfc, exp, pi, sqrt

import numpy as np
from ase.calculators.calculator import Calculator, all_changes
from ase.neighborlist import neighbor_list
from scipy.special import erfc as erfc_vec

COULOMB = 14.399645  # e^2 / (4 pi eps0) in eV * Angstrom

CHARGES = {"Ti": 2.196, "O": -1.098}

# (A [eV], rho [Angstrom], C [eV * Angstrom^6])
BUCKINGHAM = {
    ("Ti", "Ti"): (31120.1, 0.154, 5.25),
    ("Ti", "O"): (16957.53, 0.194, 12.59),
    ("O", "O"): (11782.76, 0.234, 30.22),
}


class MatsuiAkaogi(Calculator):
    """Energy and forces of the Matsui-Akaogi TiO2 potential.

    The neighbour list is built with a Verlet skin and only rebuilt when an
    atom has moved more than half of it, which keeps relaxations cheap.
    """

    implemented_properties = ["energy", "forces"]

    def __init__(self, cutoff=8.0, alpha=0.25, skin=1.0, **kwargs):
        super().__init__(**kwargs)
        self.cutoff = cutoff
        self.alpha = alpha
        self.skin = skin
        self._pairs = None
        self._ref_positions = None

    def _setup_species(self, atoms):
        symbols = atoms.get_chemical_symbols()
        unknown = set(symbols) - set(CHARGES)
        if unknown:
            raise ValueError(f"No Matsui-Akaogi parameters for {unknown}")
        species = ["Ti", "O"]
        kinds = np.array([species.index(s) for s in symbols])
        table = np.zeros((2, 2, 3))
        for (a, b), params in BUCKINGHAM.items():
            table[species.index(a), species.index(b)] = params
            table[species.index(b), species.index(a)] = params
        self._charges = np.array([CHARGES[s] for s in symbols])
        self._kinds = kinds
        self._table = table

    def _update_pairs(self, atoms):
        moved = (
            np.inf
            if self._ref_positions is None
            or len(self._ref_positions) != len(atoms)
            else np.linalg.norm(atoms.positions - self._ref_positions, axis=1).max()
        )
        if self._pairs is None or moved > 0.5 * self.skin:
            i, j, shift = neighbor_list("ijS", atoms, self.cutoff + self.skin)
            # Keep each pair once: i < j, or a self-image with a "positive"
            # lattice shift (only present in cells thinner than the cutoff).
            key = shift @ np.array([1_000_000, 1_000, 1])
            half = (i < j) | ((i == j) & (key > 0))
            self._pairs = (i[half], j[half], shift[half] @ atoms.cell.array)
            self._ref_positions = atoms.positions.copy()
            self._setup_species(atoms)

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        atoms = self.atoms
        if "numbers" in system_changes or "cell" in system_changes:
            self._pairs = None
        self._update_pairs(atoms)

        i, j, offset = self._pairs
        vec = atoms.positions[j] + offset - atoms.positions[i]
        r = np.linalg.norm(vec, axis=1)
        inside = r < self.cutoff
        i, j, vec, r = i[inside], j[inside], vec[inside], r[inside]

        a, rc = self.alpha, self.cutoff
        qq = COULOMB * self._charges[i] * self._charges[j]
        e_rc = erfc(a * rc) / rc
        f_rc = erfc(a * rc) / rc**2 + 2 * a / sqrt(pi) * exp(-(a * rc) ** 2) / rc
        erfc_r = erfc_vec(a * r)
        e_coul = qq * (erfc_r / r - e_rc + f_rc * (r - rc))
        f_coul = qq * (
            erfc_r / r**2 + 2 * a / sqrt(pi) * np.exp(-(a * r) ** 2) / r - f_rc
        )

        params = self._table[self._kinds[i], self._kinds[j]]
        amp, rho, c6 = params[:, 0], params[:, 1], params[:, 2]
        rep = amp * np.exp(-r / rho)
        e_buck = rep - c6 / r**6
        f_buck = rep / rho - 6 * c6 / r**7

        energy = np.sum(e_coul + e_buck)
        # Positive magnitude = repulsion, pushing i away from j and vice versa.
        fvec = ((f_coul + f_buck) / r)[:, None] * vec
        n = len(atoms)
        forces = np.stack(
            [
                np.bincount(j, fvec[:, k], minlength=n)
                - np.bincount(i, fvec[:, k], minlength=n)
                for k in range(3)
            ],
            axis=1,
        )

        self.results["energy"] = energy
        self.results["free_energy"] = energy
        self.results["forces"] = forces
