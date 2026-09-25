import numpy as np
import pytest

from tio2gen import build_supercell, coordination_numbers, unit_cell

AMU_PER_A3_TO_G_PER_CM3 = 1.66054


@pytest.mark.parametrize(
    "phase, n_atoms, density",
    [("anatase", 12, 3.89), ("rutile", 6, 4.25)],
)
def test_unit_cell_composition_and_density(phase, n_atoms, density):
    atoms = unit_cell(phase)
    assert len(atoms) == n_atoms
    symbols = atoms.get_chemical_symbols()
    assert symbols.count("O") == 2 * symbols.count("Ti")
    rho = atoms.get_masses().sum() / atoms.get_volume() * AMU_PER_A3_TO_G_PER_CM3
    assert rho == pytest.approx(density, abs=0.02)


@pytest.mark.parametrize("phase", ["anatase", "rutile"])
def test_ti_is_octahedrally_coordinated(phase):
    atoms = build_supercell(phase, 3)
    assert np.all(coordination_numbers(atoms) == 6)


def test_phase_name_is_case_insensitive():
    assert len(unit_cell("Rutile")) == 6


def test_unknown_phase_raises():
    with pytest.raises(ValueError, match="brookite"):
        unit_cell("brookite")


@pytest.mark.parametrize(
    "supercell, factor", [(2, 8), ((3, 2, 1), 6), ([1, 1, 4], 4)]
)
def test_supercell_size(supercell, factor):
    atoms = build_supercell("anatase", supercell)
    assert len(atoms) == 12 * factor
    assert np.allclose(
        atoms.cell.lengths(),
        unit_cell("anatase").cell.lengths() * np.broadcast_to(supercell, 3),
    )


@pytest.mark.parametrize("bad", [0, (2, 2), (1, -1, 1)])
def test_invalid_supercell_raises(bad):
    with pytest.raises(ValueError):
        build_supercell("rutile", bad)


def test_atoms_are_grouped_by_species():
    symbols = build_supercell("rutile", 2).get_chemical_symbols()
    n_ti = symbols.count("Ti")
    assert set(symbols[:n_ti]) == {"Ti"}
    assert set(symbols[n_ti:]) == {"O"}
