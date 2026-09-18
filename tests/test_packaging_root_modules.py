"""Every root-level module that shipped code imports must be declared installable.

Regression for a silent production-only failure: `gareus_report.py` lives at the
project root and is imported by `analyze_gareus_mbar.py`,
`gareus/mbar_analysis/ladder_overlap.py` and `gareus/mbar_analysis/summary.py`,
but was missing from `[tool.setuptools] py-modules`. Root-level modules are NOT
picked up by `packages.find` (which only matches `gareus*`), so it was never
installed. `gareus-analyze` therefore raised `ModuleNotFoundError: No module
named 'gareus_report'` for every run launched from any cwd except the project
root -- silently disabling the entire result-health verdict, the warning triage
and the ladder-overlap axis report, each of which is wrapped in its own
try/except and so degrades to a warning rather than failing the run.

Why the existing suite could not catch it: `[tool.pytest.ini_options]` sets
`pythonpath = ["."]`, so the project root is on sys.path for every test and
`import gareus_report` succeeds under test whether or not it is packaged. A
check that asserts on IMPORT SUCCESS passes while production stays broken. The
assertion therefore has to be on the DECLARATION LIST itself.
"""
import ast
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def _declared_py_modules() -> set[str]:
    cfg = tomllib.loads((REPO / 'pyproject.toml').read_text(encoding='utf-8'))
    return set(cfg['tool']['setuptools']['py-modules'])


def _root_level_modules() -> set[str]:
    return {p.stem for p in REPO.glob('*.py')}


def _shipped_sources(declared: set[str]) -> list[Path]:
    """Files that end up installed: the `gareus` package plus each declared
    root module. A root module importing another root module is exactly the
    edge that broke here, so the declared modules must be scanned too."""
    files = sorted(REPO.glob('gareus/**/*.py'))
    files += [REPO / f'{m}.py' for m in sorted(declared) if (REPO / f'{m}.py').exists()]
    return files


def _imported_top_level_names(path: Path) -> set[str]:
    try:
        tree = ast.parse(path.read_text(encoding='utf-8'))
    except SyntaxError:  # pragma: no cover - a broken file is another check's problem
        return set()
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(a.name.split('.')[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module.split('.')[0])
    return names


def test_every_imported_root_module_is_declared_in_py_modules():
    declared = _declared_py_modules()
    root_mods = _root_level_modules()
    offenders: dict[str, list[str]] = {}
    for src in _shipped_sources(declared):
        for name in _imported_top_level_names(src):
            if name in root_mods and name not in declared:
                offenders.setdefault(name, []).append(str(src.relative_to(REPO)))
    assert not offenders, (
        'Root-level modules imported by shipped code but absent from '
        '[tool.setuptools] py-modules in pyproject.toml -- they will not be '
        'installed, so the import fails for any run started outside the project '
        'root:\n' + '\n'.join(f'  {m}: imported by {sorted(v)}'
                              for m, v in sorted(offenders.items()))
    )


def test_gareus_report_specifically_is_declared():
    """Narrow guard on the module this regression was found through, so the
    intent survives even if the generic scan above is ever relaxed."""
    assert 'gareus_report' in _declared_py_modules()


def test_declared_py_modules_all_exist_at_the_repo_root():
    """The inverse error: declaring a module that is not there makes the wheel
    build fail late rather than at check time."""
    missing = [m for m in _declared_py_modules() if not (REPO / f'{m}.py').exists()]
    assert not missing, f'py-modules names with no matching root-level .py file: {missing}'
