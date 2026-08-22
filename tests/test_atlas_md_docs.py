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
    "guide/adaptive-workflows.md",
    "analysis/pmf-validity.md",
    "analysis/pmf-health-field-guide.md",
    "analysis/synthetic-harness.md",
    "reference/cli.md",
    "reference/configuration.md",
    "reference/output-artifacts.md",
    "tutorials/chignolin-2d.md",
    "tutorials/reproducible-fixtures.md",
    "operations/reproducibility.md",
    "operations/troubleshooting.md",
    "developer/architecture.md",
}


def test_atlas_md_has_mkdocs_configuration_and_required_pages() -> None:
    assert CONFIG.is_file(), "ATLAS-MD needs mkdocs.yml"
    config = CONFIG.read_text(encoding="utf-8")
    assert "site_name: ATLAS-MD" in config
    assert "docs_dir: docs/atlas-md" in config
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
