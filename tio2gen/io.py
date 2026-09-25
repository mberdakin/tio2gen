"""Writers for the supported output formats (XYZ, CIF and DFTB+ GEN)."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Union

from ase import Atoms
from ase.io import write

# format name -> (file extension, ASE writer format)
FORMATS: Dict[str, tuple] = {
    "xyz": ("xyz", "extxyz"),  # extended XYZ keeps the lattice vectors
    "cif": ("cif", "cif"),
    "gen": ("gen", "gen"),  # DFTB+ geometry format (periodic: 'S' type)
}


def default_basename(atoms: Atoms) -> str:
    """e.g. ``anatase_3x3x2_c0.75`` or ``rutile_4x4x6_c1.00_seed7``."""
    phase = atoms.info.get("phase", "tio2")
    supercell = atoms.info.get("supercell", "1x1x1")
    chi = atoms.info.get("crystallinity", 1.0)
    name = f"{phase}_{supercell}_c{chi:.2f}"
    if "seed" in atoms.info:
        name += f"_seed{atoms.info['seed']}"
    return name


def write_structure(
    atoms: Atoms,
    outdir: Union[str, Path] = ".",
    formats: Iterable[str] = ("xyz", "cif", "gen"),
    basename: str = None,
) -> List[Path]:
    """Write ``atoms`` in every requested format and return the file paths."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    basename = basename or default_basename(atoms)

    written = []
    for fmt in formats:
        key = fmt.lower()
        if key not in FORMATS:
            raise ValueError(
                f"Unsupported format '{fmt}'. Choose from: {', '.join(FORMATS)}"
            )
        ext, ase_format = FORMATS[key]
        path = outdir / f"{basename}.{ext}"
        if key == "xyz":
            write(path, atoms, format=ase_format)
        else:
            # CIF and GEN writers do not handle custom per-atom arrays.
            clean = atoms.copy()
            for name in list(clean.arrays):
                if name not in ("numbers", "positions"):
                    del clean.arrays[name]
            write(path, clean, format=ase_format)
        written.append(path)
    return written
