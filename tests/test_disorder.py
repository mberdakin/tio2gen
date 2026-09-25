import numpy as np
import pytest
from ase.geometry import get_distances
from ase.neighborlist import neighbor_list

from tio2gen import build_structure, build_supercell, rdf_similarity
from tio2gen.disorder import DEFAULT_MIN_DISTANCES, select_amorphous_region


def shortest_contacts(atoms):
    i, j, d = neighbor_list("ijd", atoms, 3.0)
    symbols = np.array(atoms.get_chemical_symbols())
    out = {}
    for a, b in DEFAULT_MIN_DISTANCES:
        m = (symbols[i] == a) & (symbols[j] == b)
        out[(a, b)] = d[m].min() if np.any(m) else np.inf
    return out


def test_full_crystallinity_returns_perfect_crystal():
    ref = build_supercell("anatase", 2)
    atoms = build_structure("anatase", 2, crystallinity=1.0, seed=0, relax_steps=0)
    assert np.allclose(atoms.positions, ref.positions)
    assert not atoms.arrays["amorphous"].any()


@pytest.mark.parametrize("chi", [0.0, 0.5, 0.8])
def test_amorphous_fraction_matches_crystallinity(chi):
    atoms = build_structure("rutile", (3, 3, 4), crystallinity=chi, seed=1, relax_steps=0)
    fraction = atoms.arrays["amorphous"].mean()
    assert fraction == pytest.approx(1.0 - chi, abs=1.0 / len(atoms))


@pytest.mark.parametrize("phase", ["anatase", "rutile"])
@pytest.mark.parametrize("chi", [0.0, 0.5])
def test_composition_and_cell_are_preserved(phase, chi):
    ref = build_supercell(phase, 3)
    atoms = build_structure(phase, 3, crystallinity=chi, seed=2, relax_steps=0)
    assert atoms.get_chemical_formula() == ref.get_chemical_formula()
    assert np.allclose(atoms.cell.array, ref.cell.array)


@pytest.mark.parametrize("phase", ["anatase", "rutile"])
def test_no_unphysical_contacts(phase):
    atoms = build_structure(phase, 3, crystallinity=0.0, seed=3, relax_steps=0)
    for pair, dmin in shortest_contacts(atoms).items():
        assert dmin >= DEFAULT_MIN_DISTANCES[pair] - 1e-2, pair


def test_same_seed_is_reproducible():
    a = build_structure("anatase", 2, crystallinity=0.3, seed=42, relax_steps=0)
    b = build_structure("anatase", 2, crystallinity=0.3, seed=42, relax_steps=0)
    c = build_structure("anatase", 2, crystallinity=0.3, seed=43, relax_steps=0)
    assert np.allclose(a.positions, b.positions)
    assert not np.allclose(a.positions, c.positions)


@pytest.mark.parametrize("phase", ["anatase", "rutile"])
def test_order_decreases_with_crystallinity(phase):
    ref = build_supercell(phase, 3)
    similarity = [
        rdf_similarity(build_structure(phase, 3, chi, seed=5, relax_steps=0), ref)
        for chi in (1.0, 0.75, 0.5, 0.0)
    ]
    assert similarity[0] == pytest.approx(1.0)
    assert all(np.diff(similarity) < 0)


def test_amorphous_region_is_spatially_compact():
    atoms = build_supercell("rutile", 4)
    mask = select_amorphous_region(atoms, 0.2, n_domains=1, rng=0)
    random_pick = np.random.default_rng(0).permutation(mask)

    def mean_pair_distance(selection):
        pos = atoms.positions[selection]
        _, d = get_distances(pos, cell=atoms.cell, pbc=True)
        return d.mean()

    # A single domain is a compact blob, much tighter than a random pick.
    assert mean_pair_distance(mask) < 0.8 * mean_pair_distance(random_pick)


@pytest.mark.parametrize("chi", [-0.1, 1.5])
def test_invalid_crystallinity_raises(chi):
    with pytest.raises(ValueError):
        build_structure("rutile", 2, crystallinity=chi)
