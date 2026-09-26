import numpy as np
import pytest
from ase.neighborlist import neighbor_list

from tio2gen import build_structure, build_supercell, coordination_numbers
from tio2gen.disorder import quench
from tio2gen.potential import MatsuiAkaogi


def numeric_force(atoms, index, axis, delta):
    """Central finite-difference force component (-dE/dx)."""
    position = atoms.positions[index, axis]
    energies = []
    for sign in (1, -1):
        atoms.positions[index, axis] = position + sign * delta
        energies.append(atoms.get_potential_energy())
    atoms.positions[index, axis] = position
    return (energies[1] - energies[0]) / (2 * delta)


@pytest.mark.parametrize("supercell", [2, (1, 1, 2)])  # (1, 1, 2): self-images
def test_forces_match_finite_differences(supercell):
    atoms = build_structure("rutile", supercell, 0.3, seed=1, relax_steps=0)
    atoms.calc = MatsuiAkaogi()
    forces = atoms.get_forces()
    for index in (0, len(atoms) - 1):
        numeric = [numeric_force(atoms, index, axis, 1e-4) for axis in range(3)]
        assert np.allclose(forces[index], numeric, atol=1e-4)
    assert np.allclose(forces.sum(axis=0), 0.0, atol=1e-8)


def test_rutile_is_more_stable_than_anatase():
    energies = {}
    for phase in ("anatase", "rutile"):
        atoms = build_supercell(phase, 3)
        atoms.calc = MatsuiAkaogi()
        energies[phase] = atoms.get_potential_energy() / len(atoms)
        # Experimental geometries sit close to the potential's minimum.
        assert np.abs(atoms.get_forces()).max() < 0.3
    assert energies["rutile"] < energies["anatase"]


def test_quench_restores_ti_coordination():
    atoms = build_structure("rutile", 3, 0.0, seed=4, relax_steps=0)
    before = coordination_numbers(atoms).mean()
    atoms.calc = MatsuiAkaogi()
    e_before = atoms.get_potential_energy()
    quench(atoms, steps=150)
    after = coordination_numbers(atoms).mean()
    atoms.calc = MatsuiAkaogi()
    assert atoms.get_potential_energy() < e_before
    assert after > before + 0.5


def test_quenched_structure_has_no_hard_core_pile_up():
    atoms = build_structure("anatase", (3, 3, 2), 0.0, seed=2, relax_steps=200)
    d = neighbor_list("d", atoms, 2.3)
    # Without the quench ~all short contacts sit exactly at 1.70 A.
    assert np.mean(np.abs(d - 1.70) < 0.02) < 0.02
