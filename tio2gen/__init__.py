"""tio2gen: TiO2 supercells with tunable crystallinity, built on ASE."""

__version__ = "0.1.0"

from .analysis import coordination_numbers, radial_distribution, rdf_similarity
from .builder import build_structure, build_supercell
from .disorder import apply_disorder
from .io import write_structure
from .phases import available_phases, unit_cell

__all__ = [
    "apply_disorder",
    "available_phases",
    "build_structure",
    "build_supercell",
    "coordination_numbers",
    "radial_distribution",
    "rdf_similarity",
    "unit_cell",
    "write_structure",
]
