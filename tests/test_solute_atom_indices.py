from gareus.cv import solute_atom_indices


class _Atom:
    def __init__(self, index):
        self.index = index


class _Res:
    def __init__(self, name, atom_indices):
        self.name = name
        self._atoms = [_Atom(i) for i in atom_indices]

    def atoms(self):
        return iter(self._atoms)


class _Topo:
    def __init__(self, residues):
        self._res = residues

    def residues(self):
        return iter(self._res)


def test_selects_solute_excludes_water_and_ions():
    # ACE/ALA/NME (indices 0-21) + water (HOH) + ion (NA)
    topo = _Topo([
        _Res("ACE", [0, 1, 2, 3, 4, 5]),
        _Res("ALA", list(range(6, 16))),
        _Res("NME", list(range(16, 22))),
        _Res("HOH", [22, 23, 24]),
        _Res("NA", [25]),
    ])
    assert solute_atom_indices(topo) == list(range(22))


def test_ascending_order_regardless_of_residue_order():
    topo = _Topo([
        _Res("NME", [16, 17]),
        _Res("ACE", [0, 1]),
        _Res("HOH", [99]),
    ])
    assert solute_atom_indices(topo) == [0, 1, 16, 17]


def test_empty_when_only_solvent():
    topo = _Topo([_Res("HOH", [0, 1, 2]), _Res("CL", [3])])
    assert solute_atom_indices(topo) == []
