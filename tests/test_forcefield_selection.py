"""Force-field and water-model selection, and the pairing guard.

The protein force field used to be hardcoded -- `make_forcefield` returned
`app.ForceField("amber14-all.xml", water_xml)` with no way to choose -- and OPC
was absent from the water map, so ff19SB/OPC could not be run at all.

Two hazards shape this design:

* **A duplicate map.** `system_setup.make_forcefield` and
  `provenance._forcefield_settings` each carried their own copy of the water
  dict. Updating one and not the other makes the provenance record disagree
  with what actually ran, silently. Both now call `forcefield_xml_paths`.

* **Resume rebuilding with a different force field.** `checkpoints.py` and
  `swarm/driver.py` construct the force field independently of the production
  path. A per-call default would let a resume rebuild an ff19SB system as
  ff14SB without complaint, which is unrecoverable and invisible. Selection is
  therefore derived from `args` through one function everywhere.

ff19SB was parameterized against OPC; pairing it with TIP3P is a known
mismatch, so it is refused unless explicitly overridden.
"""

from pathlib import Path
import argparse
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gareus.system_setup import forcefield_xml_paths


def test_default_pairing_is_unchanged_ff14sb_tip3p():
    """Every existing config must keep loading exactly what it loaded before."""
    assert forcefield_xml_paths("ff14SB", "tip3p") == ["amber14-all.xml", "amber14/tip3p.xml"]


def test_ff19sb_selects_the_amber19_protein_xml():
    assert forcefield_xml_paths("ff19SB", "opc") == ["amber19-all.xml", "amber14/opc.xml"]


def test_opc_is_available_as_a_water_model():
    assert forcefield_xml_paths("ff14SB", "opc")[1] == "amber14/opc.xml"


def test_every_previously_supported_water_model_still_resolves():
    for water in ("tip3p", "tip3pfb", "spce", "tip4pew"):
        paths = forcefield_xml_paths("ff14SB", water)
        assert paths[0] == "amber14-all.xml"
        assert paths[1].endswith(f"{water}.xml")


def test_ff19sb_with_tip3p_is_refused():
    """ff19SB was parameterized against OPC. Silently allowing TIP3P would
    produce a run that looks fine and is wrong."""
    with pytest.raises(ValueError, match="ff19SB"):
        forcefield_xml_paths("ff19SB", "tip3p")


def test_ff19sb_mismatch_can_be_overridden_deliberately():
    paths = forcefield_xml_paths("ff19SB", "tip3p", allow_mismatch=True)
    assert paths == ["amber19-all.xml", "amber14/tip3p.xml"]


def test_unknown_forcefield_is_refused_not_defaulted():
    with pytest.raises(ValueError):
        forcefield_xml_paths("ff99SB", "tip3p")


def test_unknown_water_model_is_refused_not_defaulted():
    with pytest.raises(ValueError):
        forcefield_xml_paths("ff14SB", "tip5p")


# --- args plumbing: one source of truth, no per-call defaults ----------------

from gareus.system_setup import forcefield_selection_from_args


def test_selection_reads_args_and_defaults_to_the_previous_behaviour():
    ff, water, allow = forcefield_selection_from_args(argparse.Namespace())
    assert (ff, water, allow) == ("ff14SB", "tip3p", False)


def test_selection_reads_an_explicit_ff19sb_opc_run():
    ff, water, allow = forcefield_selection_from_args(
        argparse.Namespace(forcefield="ff19SB", water_model="opc"))
    assert (ff, water) == ("ff19SB", "opc")


# --- provenance must report what actually ran -------------------------------

from gareus.provenance import _forcefield_settings


def test_provenance_records_the_ff19sb_opc_pair():
    s = _forcefield_settings(argparse.Namespace(forcefield="ff19SB", water_model="opc"))
    assert s["forcefield_xml"] == ["amber19-all.xml", "amber14/opc.xml"]
    assert s["forcefield"] == "ff19SB"
    assert s["water_model"] == "opc"


def test_provenance_default_matches_the_historical_record():
    s = _forcefield_settings(argparse.Namespace())
    assert s["forcefield_xml"] == ["amber14-all.xml", "amber14/tip3p.xml"]
    assert s["forcefield"] == "ff14SB"


def test_provenance_and_system_setup_cannot_disagree():
    """The duplicate-map hazard: both must come from one function."""
    args = argparse.Namespace(forcefield="ff19SB", water_model="opc")
    ff, water, allow = forcefield_selection_from_args(args)
    assert _forcefield_settings(args)["forcefield_xml"] == forcefield_xml_paths(ff, water, allow_mismatch=allow)
