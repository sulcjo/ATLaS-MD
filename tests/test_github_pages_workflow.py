"""Contract for ATLAS-MD GitHub Pages deployment."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "deploy-atlas-md.yml"


def test_atlas_md_pages_workflow_builds_and_deploys_from_main() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    for required in (
        "branches: [main]",
        "python -m mkdocs build --strict --site-dir site",
        "actions/upload-pages-artifact@v4",
        "actions/deploy-pages@v4",
        "pages: write",
        "id-token: write",
        "name: github-pages",
    ):
        assert required in text
