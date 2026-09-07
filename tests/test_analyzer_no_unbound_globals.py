"""Every name a function in analyze_gareus_mbar.py reads as a global must exist at module scope
(or be a builtin).  A block that references another function's local (``_sp_stats`` in
``analyze_distance_rg_2d_fes``, 2026-09-07) only fails at run time, after the MBAR solve and
long trajectory passes -- the S3 pilot analysis crashed there and never wrote pmf_summary.json."""
import builtins
import symtable
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "analyze_gareus_mbar.py"
_MODULE_DUNDERS = {"__file__", "__name__", "__doc__", "__spec__", "__package__", "__loader__", "__builtins__"}


def _unbound_global_reads(path: Path) -> list[tuple[str, str]]:
    src = path.read_text()
    top = symtable.symtable(src, str(path), "exec")
    module_names = {s.get_name() for s in top.get_symbols()}
    # names bound anywhere at module level count (imports, defs, assignments, star-imports aside)
    problems = []

    def walk(tab, owner):
        for sym in tab.get_symbols():
            if (tab.get_type() == "function" and sym.is_global() and sym.is_referenced()
                    and not sym.is_assigned() and sym.get_name() not in module_names
                    and not hasattr(builtins, sym.get_name())
                    and sym.get_name() not in _MODULE_DUNDERS):
                problems.append((owner, sym.get_name()))
        for child in tab.get_children():
            walk(child, f"{owner}.{child.get_name()}" if owner else child.get_name())

    walk(top, "")
    return sorted(set(problems))


def test_analyzer_functions_read_only_defined_globals():
    problems = _unbound_global_reads(SCRIPT)
    assert not problems, f"functions read undefined globals: {problems}"
