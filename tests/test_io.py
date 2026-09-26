import numpy as np
import pytest
from ase.io import read

from tio2gen import build_structure, write_structure
from tio2gen.cli import main


@pytest.mark.parametrize("fmt", ["xyz", "cif", "gen"])
def test_roundtrip(tmp_path, fmt):
    atoms = build_structure("anatase", 2, crystallinity=0.5, seed=0, relax_steps=0)
    (path,) = write_structure(atoms, tmp_path, formats=[fmt])
    back = read(path)
    assert back.get_chemical_formula() == atoms.get_chemical_formula()
    assert np.allclose(back.cell.array, atoms.cell.array, atol=1e-4)
    assert back.pbc.all()


def test_gen_file_is_periodic(tmp_path):
    atoms = build_structure("rutile", 2)
    (path,) = write_structure(atoms, tmp_path, formats=["gen"])
    n_atoms, kind = path.read_text().split()[:2]
    assert int(n_atoms) == len(atoms)
    assert kind == "S"


def test_xyz_keeps_amorphous_flags(tmp_path):
    atoms = build_structure("rutile", 2, crystallinity=0.5, seed=1, relax_steps=0)
    (path,) = write_structure(atoms, tmp_path, formats=["xyz"])
    back = read(path)
    assert np.array_equal(back.arrays["amorphous"], atoms.arrays["amorphous"])
    assert back.info["crystallinity"] == pytest.approx(0.5)


def test_default_filenames(tmp_path):
    atoms = build_structure("rutile", (2, 2, 3), crystallinity=0.25, seed=7, relax_steps=0)
    paths = write_structure(atoms, tmp_path)
    assert sorted(p.name for p in paths) == [
        "rutile_2x2x3_c0.25_seed7.cif",
        "rutile_2x2x3_c0.25_seed7.gen",
        "rutile_2x2x3_c0.25_seed7.xyz",
    ]


def test_unknown_format_raises(tmp_path):
    with pytest.raises(ValueError):
        write_structure(build_structure("rutile"), tmp_path, formats=["pdb"])


@pytest.mark.filterwarnings("ignore:Quench did not converge")
def test_cli_writes_one_structure_per_crystallinity(tmp_path, capsys):
    code = main([
        "--phase", "rutile", "--supercell", "2", "2", "3",
        "-c", "1", "0.5", "--seed", "3", "--relax-steps", "20",
        "-f", "gen", "-o", str(tmp_path),
    ])
    assert code == 0
    names = sorted(p.name for p in tmp_path.iterdir())
    assert names == ["rutile_2x2x3_c0.50_seed3.gen", "rutile_2x2x3_c1.00_seed3.gen"]
    assert "Ti24O48" in capsys.readouterr().out
