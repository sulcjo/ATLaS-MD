"""Contract checks for source-controlled ATLAS-MD documentation."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs" / "atlas-md"
CONFIG = ROOT / "mkdocs.yml"

REQUIRED_PAGES = {
    "index.md",
    "start/installation.md",
    "start/quickstart.md",
    "guide/collective-variables.md",
    "guide/foundations.md",
    "guide/gamd-calibration.md",
    "guide/adaptive-workflows.md",
    "analysis/pmf-validity.md",
    "analysis/reweighting.md",
    "analysis/validation-workflow.md",
    "analysis/pmf-health-field-guide.md",
    "analysis/synthetic-harness.md",
    "reference/cli.md",
    "reference/configuration.md",
    "reference/output-artifacts.md",
    "tutorials/chignolin-2d.md",
    "tutorials/genpept-chignolin-case-study.md",
    "tutorials/reproducible-fixtures.md",
    "operations/reproducibility.md",
    "operations/reporting-checklist.md",
    "operations/troubleshooting.md",
    "developer/architecture.md",
    "start/symbols-and-citations.md",
}


def test_atlas_md_has_mkdocs_configuration_and_required_pages() -> None:
    assert CONFIG.is_file(), "ATLAS-MD needs mkdocs.yml"
    config = CONFIG.read_text(encoding="utf-8")
    assert "site_name: ATLAS-MD" in config
    assert "docs_dir: docs/atlas-md" in config
    assert "pymdownx.arithmatex" in config
    assert "assets/mathjax.js" in config
    missing = sorted(page for page in REQUIRED_PAGES if not (DOCS / page).is_file())
    assert not missing, f"Missing ATLAS-MD pages: {missing}"


def test_atlas_md_local_markdown_links_resolve() -> None:
    missing: list[str] = []
    link_pattern = re.compile(r"!?\[[^]]*\]\(([^)]+)\)")
    for page in DOCS.rglob("*.md"):
        for target in link_pattern.findall(page.read_text(encoding="utf-8")):
            target = target.split("#", 1)[0].strip()
            if not target or "://" in target or target.startswith("mailto:"):
                continue
            destination = (page.parent / target).resolve()
            if not destination.exists():
                missing.append(f"{page.relative_to(ROOT)} -> {target}")
    assert not missing, "Broken local documentation links:\n" + "\n".join(missing)


def test_atlas_md_source_is_not_ignored_by_git() -> None:
    result = subprocess.run(
        ["git", "check-ignore", "docs/atlas-md/index.md"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1, result.stdout


def test_atlas_md_documents_rendered_diagrams_and_synthetic_harness() -> None:
    config = CONFIG.read_text(encoding="utf-8")
    index = (DOCS / "index.md").read_text(encoding="utf-8")
    synth = (DOCS / "analysis" / "synthetic-harness.md").read_text(encoding="utf-8")
    assert "class: mermaid" in config
    assert index.count("```mermaid") >= 1
    assert "```mermaid" in synth
    assert "run_adaptive_feedback_dispatcher_2d" in synth
    assert "propose_actions_from_diagnostics" in synth


def test_atlas_md_has_a_schema_accurate_pmf_field_guide_and_tested_fixtures() -> None:
    guide = (DOCS / "analysis" / "pmf-health-field-guide.md").read_text(encoding="utf-8")
    fixtures = (DOCS / "tutorials" / "reproducible-fixtures.md").read_text(encoding="utf-8")
    for key in ("pmf_summary.json", "base_ess", "neighbor_overlap", "health"):
        assert key in guide
    assert "gareus-test-run --dry-run" in fixtures
    assert "python -m gareus.synth" in fixtures


def test_genpept_chignolin_case_study_has_pipeline_evidence_and_limits() -> None:
    page = (DOCS / "tutorials" / "genpept-chignolin-case-study.md").read_text(
        encoding="utf-8"
    )
    config = CONFIG.read_text(encoding="utf-8")
    assets = DOCS / "assets" / "genpept-chignolin"

    assert "GENPEPT Chignolin case study" in config
    assert "not an equilibrium free-energy surface" in page
    for stage in (
        "Ramachandran-state generation",
        "implicit minimization",
        "basin hopping",
        "ANM/NMA expansion",
        "PCA-frontier expansion",
        "final survivor selection",
        "GAREUS handoff",
    ):
        assert stage.lower() in page.lower()
    for control in (
        "run02_raw_generation_only",
        "run03_bh_only",
        "run04_nma_only",
        "run05_minimal_full",
        "run06_pca_frontier_control",
        "run07_reasonable_all_methods",
        "chignolin_genpept_r7",
    ):
        assert control in page
    for figure in (
        "run07-space-score-growth.png",
        "run07-pca-area-growth.png",
        "r7-stage-source-composition.png",
        "r7-end-to-end-pseudo-fes.png",
    ):
        assert (assets / figure).is_file(), figure
    assert "Atomistic explicit minimization: disabled" in page
    assert "Optional explicit minimization" not in page
    assert "BH enabled: NMA expands BH minima" in page
