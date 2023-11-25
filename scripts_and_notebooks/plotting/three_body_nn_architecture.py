import tensorflow as tf
from mbeml.nn_models_functional import build_three_body_model


if __name__ == "__main__":
    model = build_three_body_model(
        two_body_units=[64, 64],
        three_body_units=[64, 64],
        num_ligand_features=32,
        num_outputs=4,
    )
    tf.keras.utils.plot_model(
        model,
        to_file="plots/three_body_nn_architecture.png",
        dpi=200,
        expand_nested=True,
        show_shapes=True,
        show_layer_activations=True,
        show_layer_names=True,
    )
