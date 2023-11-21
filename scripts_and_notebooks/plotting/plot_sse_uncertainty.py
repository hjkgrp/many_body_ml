experiments = [
    Experiment(name="krr_standard_racs", features=LigandFeatures.STANDARD_RACS),
    Experiment(name="krr_two_body", features=LigandFeatures.LIGAND_RACS),
    Experiment(name="krr_three_body", features=LigandFeatures.LIGAND_RACS),
    Experiment(
        name="nn_standard_racs", features=LigandFeatures.STANDARD_RACS, is_nn=True
    ),
    Experiment(name="nn_two_body", features=LigandFeatures.LIGAND_RACS, is_nn=True),
    Experiment(name="nn_three_body", features=LigandFeatures.LIGAND_RACS, is_nn=True),
]
