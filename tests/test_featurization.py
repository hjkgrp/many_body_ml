from mbeml.featurization import generate_standard_racs_names, generate_ligand_racs_names


def test_standard_racs():
    names = generate_standard_racs_names()
    assert len(names) == 184

    names = generate_standard_racs_names(remove_trivial=True)
    assert len(names) == 155


def test_ligand_racs():
    names = generate_ligand_racs_names()
    assert len(names) == 33 * 6

    names = generate_ligand_racs_names(remove_trivial=True)
    assert len(names) == 32 * 6
