import tensorflow as tf
from typing import List
from mbeml.nn_layers import (
    AddSpinEncoding,
    TwoBodyPrep,
    ThreeBodyPrep,
    ThreeBodyPrepFeatureSym,
    ThreeBodyMask,
)
from mbeml.constants import cis_pairs, trans_pairs


def build_mlp(
    hidden_units=(64, 32),
    num_outputs=1,
    activation="softplus",
    kernel_regularizer=None,
    dense_kw=None,
    final_kw=None,
    dropout_rate: float = 0.0,
    name: str = "mlp",
):
    layers = []
    if kernel_regularizer is None:
        kernel_regularizer = tf.keras.regularizers.L2()
    if dense_kw is None:
        dense_kw = {}
    if final_kw is None:
        final_kw = {}
    for u in hidden_units:
        layers.append(
            tf.keras.layers.Dense(
                u,
                activation=activation,
                kernel_regularizer=kernel_regularizer,
                **dense_kw,
            )
        )
        layers.append(tf.keras.layers.Dropout(dropout_rate))
    layers.append(tf.keras.layers.Dense(num_outputs, **final_kw))

    return tf.keras.Sequential(layers=layers, name=name)


def build_nn_ensemble(models: List[tf.keras.Model], return_std=False) -> tf.keras.Model:
    """_summary_

    Parameters
    ----------
    models : List[tf.keras.Model]
        _description_
    return_std : bool, optional
        _description_, by default False

    Returns
    -------
    tf.keras.Model
        _description_
    """
    input_shapes = models[0].input_shape
    # Slice [1:] to remove batch size, following documentation of keras.Input
    core_inp = tf.keras.Input(shape=input_shapes["core"][1:], name="core_input")
    ligands_inp = tf.keras.Input(
        shape=input_shapes["ligands"][1:], name="ligands_input"
    )

    stacked_output = tf.stack(
        [model({"core": core_inp, "ligands": ligands_inp}) for model in models],
        axis=0,
        name="stacked_outputs",
    )
    if return_std:
        outputs = tf.math.reduce_mean(stacked_output, axis=0), tf.math.reduce_std(
            stacked_output, axis=0
        )
    else:
        outputs = tf.math.reduce_mean(stacked_output, axis=0, name="mean")

    model = tf.keras.models.Model(
        inputs={"core": core_inp, "ligands": ligands_inp}, outputs=outputs
    )
    return model


def build_two_body_model(
    two_body_units: List[int] = [16, 8],
    l2: float = 0.01,
    dropout_rate: float = 0.0,
    racs_norm=None,
    output_norm=None,
    spin_dependent: bool = False,
    num_core_features=7,
    num_ligands=6,
    num_ligand_features=184,
    num_outputs: int = 1,
):
    core_inp = tf.keras.Input(shape=(num_core_features,), name="core_input")
    ligands_inp = tf.keras.Input(
        shape=(num_ligands, num_ligand_features), name="ligands_input"
    )
    if racs_norm is not None:
        ligands_normed = racs_norm(ligands_inp)
    else:
        ligands_normed = ligands_inp
    two_body_prep = TwoBodyPrep()
    nn = build_mlp(
        hidden_units=two_body_units,
        num_outputs=num_outputs,
        kernel_regularizer=tf.keras.regularizers.L2(l2),
        dropout_rate=dropout_rate,
        name="two_body_nn",
    )

    if spin_dependent:
        core_spins = AddSpinEncoding()(core_inp)
        combined_inputs = [
            two_body_prep([core_spin, ligands_normed]) for core_spin in core_spins
        ]
        output = tf.concat([nn(inp) for inp in combined_inputs], axis=-1)
    else:
        combined_input = two_body_prep([core_inp, ligands_normed])
        output = nn(combined_input)
    # Sum over the six ligands
    output = tf.reduce_sum(output, axis=-2)
    if output_norm is not None:
        output = output_norm(output, invert=True)

    model = tf.keras.models.Model(
        inputs={"core": core_inp, "ligands": ligands_inp}, outputs=output
    )
    return model


def build_three_body_model(
    two_body_units=[16, 8],
    three_body_units=[16, 8],
    l2=0.01,
    dropout_rate=0.0,
    racs_norm=None,
    output_norm=None,
    spin_dependent=False,
    masked=False,
    features_sym=True,
    one_body_term=False,
    two_body_terms=True,
    num_core_features=7,
    num_ligands=6,
    num_ligand_features=33,
    num_outputs=1,
):
    core_inp = tf.keras.Input(shape=(num_core_features,), name="core_input")
    ligands_inp = tf.keras.Input(
        shape=(num_ligands, num_ligand_features), name="ligands_input"
    )
    if racs_norm is not None:
        ligands_normed = racs_norm(ligands_inp)
    else:
        ligands_normed = ligands_inp
    # One body term:
    if one_body_term:
        one_body_nn = build_mlp(
            hidden_units=[],
            num_outputs=num_outputs,
            name="one_body_nn",
        )
        if spin_dependent:
            raise NotImplementedError(
                "Combination of spin dependent and one body term is not implemented."
            )
    # Two body layers
    if two_body_terms:
        two_body_prep = TwoBodyPrep()
        two_body_nn = build_mlp(
            hidden_units=two_body_units,
            num_outputs=num_outputs,
            kernel_regularizer=tf.keras.regularizers.L2(l2),
            dropout_rate=dropout_rate,
            name="two_body_nn",
        )
    # Three body layers cis
    if features_sym:
        three_body_prep_cis = ThreeBodyPrepFeatureSym(
            pairs=cis_pairs, name="three_body_prep_cis"
        )
    else:
        three_body_prep_cis = ThreeBodyPrep(pairs=cis_pairs, name="three_body_prep_cis")
    three_body_nn_cis = build_mlp(
        hidden_units=three_body_units,
        num_outputs=num_outputs,
        kernel_regularizer=tf.keras.regularizers.L2(l2),
        dropout_rate=dropout_rate,
        name="three_body_nn_cis",
    )
    # Three body layers trans
    if features_sym:
        three_body_prep_trans = ThreeBodyPrepFeatureSym(
            pairs=trans_pairs, name="three_body_prep_trans"
        )
    else:
        three_body_prep_trans = ThreeBodyPrep(
            pairs=trans_pairs, name="three_body_prep_trans"
        )
    three_body_nn_trans = build_mlp(
        hidden_units=three_body_units,
        num_outputs=num_outputs,
        kernel_regularizer=tf.keras.regularizers.L2(l2),
        dropout_rate=dropout_rate,
        name="three_body_nn_trans",
    )
    if masked:
        three_body_mask_cis = ThreeBodyMask(pairs=cis_pairs, name="three_body_mask_cis")
        three_body_mask_trans = ThreeBodyMask(
            pairs=trans_pairs, name="three_body_mask_trans"
        )

    if spin_dependent:
        core_spins = AddSpinEncoding()(core_inp)
        # Two body interactions
        if two_body_terms:
            two_body_inputs = [
                two_body_prep([core_spin, ligands_normed]) for core_spin in core_spins
            ]
            two_body_output = tf.concat(
                [two_body_nn(inp) for inp in two_body_inputs], axis=-1
            )
        # Three body cis
        three_body_inputs_cis = [
            three_body_prep_cis([core_spin, ligands_normed]) for core_spin in core_spins
        ]
        three_body_output_cis = tf.concat(
            [three_body_nn_cis(inp) for inp in three_body_inputs_cis], axis=-1
        )
        # Three body trans
        three_body_inputs_trans = [
            three_body_prep_trans([core_spin, ligands_normed])
            for core_spin in core_spins
        ]
        three_body_output_trans = tf.concat(
            [three_body_nn_trans(inp) for inp in three_body_inputs_trans], axis=-1
        )
        # Necessary for masking: Assign LS as default spin
        three_body_input_cis = three_body_inputs_cis[0]
        three_body_input_trans = three_body_inputs_trans[0]
    else:
        if one_body_term:
            one_body_output = one_body_nn(core_inp)
        # Two body interactions
        if two_body_terms:
            two_body_input = two_body_prep([core_inp, ligands_normed])
            two_body_output = two_body_nn(two_body_input)
        # Three body cis
        three_body_input_cis = three_body_prep_cis([core_inp, ligands_normed])
        three_body_output_cis = three_body_nn_cis(three_body_input_cis)
        # Three body trans
        three_body_input_trans = three_body_prep_trans([core_inp, ligands_normed])
        three_body_output_trans = three_body_nn_trans(three_body_input_trans)

    # Sum over the two body interactions
    if two_body_terms:
        two_body_output = tf.reduce_sum(two_body_output, axis=-2)
    if not features_sym:
        # Average over the two permutations
        three_body_output_cis = tf.reduce_mean(three_body_output_cis, axis=-2)
        three_body_output_trans = tf.reduce_mean(three_body_output_trans, axis=-2)
    if masked:
        three_body_output_cis = tf.keras.layers.Multiply(name="masking_layer_cis")(
            [three_body_output_cis, three_body_mask_cis([core_inp, ligands_normed])]
        )
        three_body_output_trans = tf.keras.layers.Multiply(name="masking_layer_trans")(
            [three_body_output_trans, three_body_mask_trans([core_inp, ligands_normed])]
        )
    # Sum over the three body interactions
    three_body_output_cis = tf.reduce_sum(three_body_output_cis, axis=-2)
    three_body_output_trans = tf.reduce_sum(three_body_output_trans, axis=-2)

    # Add all three outputs
    mbe_terms = []
    if one_body_term:
        mbe_terms.append(one_body_output)
    if two_body_terms:
        mbe_terms.append(two_body_output)
    mbe_terms.extend([three_body_output_cis, three_body_output_trans])
    output = tf.keras.layers.Add()(mbe_terms)

    if output_norm is not None:
        output = output_norm(output, invert=True)

    model = tf.keras.models.Model(
        inputs={"core": core_inp, "ligands": ligands_inp}, outputs=output
    )
    return model


def build_standard_racs_model(
    hidden_units=[16, 8],
    l2=0.01,
    dropout_rate=0.0,
    racs_norm=None,
    output_norm=None,
    spin_dependent=False,
    num_core_features=7,
    num_ligand_features=184,
    num_outputs=1,
):
    core_inp = tf.keras.Input(shape=(num_core_features,), name="core_input")
    ligands_inp = tf.keras.Input(shape=(num_ligand_features,), name="ligands_input")
    if racs_norm is not None:
        ligands_normed = racs_norm(ligands_inp)
    else:
        ligands_normed = ligands_inp
    nn = build_mlp(
        hidden_units=hidden_units,
        num_outputs=num_outputs,
        kernel_regularizer=tf.keras.regularizers.L2(l2),
        dropout_rate=dropout_rate,
    )
    if spin_dependent:
        core_spins = AddSpinEncoding()(core_inp)
        combined_inputs = [
            tf.concat([core_spin, ligands_normed], axis=-1) for core_spin in core_spins
        ]
        output = tf.concat([nn(inp) for inp in combined_inputs], axis=-1)
    else:
        combined_input = tf.concat([core_inp, ligands_normed], axis=-1)
        output = nn(combined_input)
    if output_norm is not None:
        output = output_norm(output, invert=True)

    model = tf.keras.models.Model(
        inputs={"core": core_inp, "ligands": ligands_inp}, outputs=output
    )
    return model
