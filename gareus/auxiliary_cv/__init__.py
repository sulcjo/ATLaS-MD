"""Auxiliary collective-variable states (spec 2026-10-07-auxiliary-cv-gibbs-production-spec)."""
from .model import AUX_MODEL_SCHEMA, AuxModel, AuxModelError

__all__ = ["AUX_MODEL_SCHEMA", "AuxModel", "AuxModelError"]

from .evaluate import aux_energy_kj, aux_forces_kj_nm, z_and_gradient, z_from_dihedrals, z_from_positions
from .features import AuxGeometryError, check_feature_atoms, openmm_dihedrals

__all__ += ["aux_energy_kj", "aux_forces_kj_nm", "z_and_gradient", "z_from_dihedrals",
            "z_from_positions", "AuxGeometryError", "check_feature_atoms", "openmm_dihedrals"]

from .force import AUX_FORCE_NAME, AuxForceInfo, build_aux_force, set_aux_parameters

__all__ += ["AUX_FORCE_NAME", "AuxForceInfo", "build_aux_force", "set_aux_parameters"]

from .sample_schema import (AUX_SAMPLES_SCHEMA, AUX_SAMPLES_SCHEMA_V2, AuxObservation, AuxSampleSchema,
                            build_sample_schema, observe_carrier)

__all__ += ["AUX_SAMPLES_SCHEMA", "AUX_SAMPLES_SCHEMA_V2", "AuxObservation", "AuxSampleSchema",
            "build_sample_schema", "observe_carrier"]

from .sidechain_model import SidechainModel, SidechainModelError

__all__ += ["SidechainModel", "SidechainModelError"]
