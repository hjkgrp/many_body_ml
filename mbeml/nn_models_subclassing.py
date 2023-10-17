import tensorflow as tf


class MLP(tf.keras.layers.Layer):
    def __init__(
        self,
        hidden_units=(64, 32),
        activation="softplus",
        kernel_regularizer=tf.keras.regularizers.L2(),
        dense_kw=None,
        final_kw=None,
        dropout_rate=0.0,
        **kwargs,
    ):
        super(MLP, self).__init__(**kwargs)
        self.layers = []
        if dense_kw is None:
            dense_kw = {}
        if final_kw is None:
            final_kw = {}
        for u in hidden_units:
            self.layers.append(
                tf.keras.layers.Dense(
                    u,
                    activation=activation,
                    kernel_regularizer=kernel_regularizer,
                    **dense_kw,
                )
            )
            self.layers.append(tf.keras.layers.Dropout(dropout_rate))
        self.layers.append(tf.keras.layers.Dense(1, **final_kw))

    @tf.function
    def call(self, inputs, training=None):
        output = self.layers[0](inputs)
        for layer in self.layers[1:]:
            output = layer(output, training=training)
        return output


class TwoBodyLayer(tf.keras.layers.Layer):
    def __init__(self, mlp_kw=None, **kwargs):
        super(TwoBodyLayer, self).__init__(**kwargs)
        self.nn = MLP(**mlp_kw)

    @tf.function
    def call(self, core, ligs, training=None):
        inp = tf.concat([tf.repeat(core[:, tf.newaxis, :], 6, axis=1), ligs], axis=-1)
        # Shape = (None, 6, 1)
        res = self.nn(inp, training=training)
        # Sum over the ligands
        return tf.reduce_sum(res, axis=-2)


class ThreeBodyLayer(tf.keras.layers.Layer):
    def __init__(self, pairs, mlp_kw=None, **kwargs):
        super(ThreeBodyLayer, self).__init__(**kwargs)
        self.nn = MLP(**mlp_kw)
        self.pairs = pairs

    @tf.function
    def call(self, core, ligs, training=None):
        output = []
        for i, j in self.pairs:
            inp1 = tf.concat([core, ligs[:, i, :], ligs[:, j, :]], axis=-1)
            inp2 = tf.concat([core, ligs[:, j, :], ligs[:, i, :]], axis=-1)
            res = 0.5 * (
                self.nn(inp1, training=training) + self.nn(inp2, training=training)
            )
            # cond = tf.reduce_all(tf.math.abs(inp1 - inp2) < 1e-6,
            #                      axis=-1, keepdims=True)
            # output.append(tf.where(cond, tf.zeros_like(res), res))
            output.append(res)
        # Stack and sum over the pairs
        return tf.reduce_sum(tf.stack(output, axis=1), axis=-2)


class TwoBodyModel(tf.keras.Model):
    def __init__(
        self,
        two_body_units=[16, 8],
        l2=0.01,
        dropout_rate=0.0,
        spin_dependent=False,
        racs_norm=None,
        output_norm=None,
    ):
        super(TwoBodyModel, self).__init__()

        self.spin_dependent = spin_dependent
        self.racs_norm = racs_norm
        self.output_norm = output_norm

        self.two_body_nn = TwoBodyLayer(
            mlp_kw=dict(
                hidden_units=two_body_units,
                kernel_regularizer=tf.keras.regularizers.L2(l2),
                dropout_rate=dropout_rate,
            ),
            name="two_body_nn",
        )

    @tf.function
    def call(self, inputs, training=None):
        core = inputs["core"]
        ligs = self.racs_norm(inputs["ligands"])
        if self.spin_dependent:
            core_ls = tf.concat(
                [
                    core,
                    tf.ones((tf.shape(core)[0], 1)),
                    tf.zeros((tf.shape(core)[0], 1)),
                ],
                axis=-1,
            )
            core_hs = tf.concat(
                [
                    core,
                    tf.zeros((tf.shape(core)[0], 1)),
                    tf.ones((tf.shape(core)[0], 1)),
                ],
                axis=-1,
            )
            return self.output_norm(
                tf.concat(
                    [
                        self.two_body_nn(core_ls, ligs, training=training),
                        self.two_body_nn(core_hs, ligs, training=training),
                    ],
                    axis=-1,
                ),
                invert=True,
            )
        return self.output_norm(
            self.two_body_nn(core, ligs, training=training), invert=True
        )


class ThreeBodyModel(tf.keras.Model):
    cis_pairs = [
        (0, 1),
        (0, 3),
        (0, 4),
        (0, 5),
        (1, 2),
        (1, 4),
        (1, 5),
        (2, 3),
        (2, 4),
        (2, 5),
        (3, 4),
        (3, 5),
    ]
    trans_pairs = [(0, 2), (1, 3), (4, 5)]

    def __init__(
        self,
        two_body_units=[16, 8],
        three_body_units=[16, 8],
        l2=0.01,
        dropout_rate=0.0,
        spin_dependent=False,
        racs_norm=None,
        output_norm=None,
    ):
        super(ThreeBodyModel, self).__init__()

        self.spin_dependent = spin_dependent
        self.racs_norm = racs_norm
        self.output_norm = output_norm

        self.two_body_nn = TwoBodyLayer(
            mlp_kw=dict(
                hidden_units=two_body_units,
                kernel_regularizer=tf.keras.regularizers.L2(l2),
                dropout_rate=dropout_rate,
            ),
            name="two_body_nn",
        )
        self.three_body_nn_cis = ThreeBodyLayer(
            mlp_kw=dict(
                hidden_units=three_body_units,
                kernel_regularizer=tf.keras.regularizers.L2(l2),
                dropout_rate=dropout_rate,
            ),
            pairs=self.cis_pairs,
            name="three_body_nn_cis",
        )
        self.three_body_nn_trans = ThreeBodyLayer(
            mlp_kw=dict(
                hidden_units=three_body_units,
                kernel_regularizer=tf.keras.regularizers.L2(l2),
                dropout_rate=dropout_rate,
            ),
            pairs=self.trans_pairs,
            name="three_body_nn_trans",
        )

    @tf.function
    def call(self, inputs, training=None):
        core = inputs["core"]
        ligs = self.racs_norm(inputs["ligands"])
        if self.spin_dependent:
            core_ls = tf.concat(
                [
                    core,
                    tf.ones((tf.shape(core)[0], 1)),
                    tf.zeros((tf.shape(core)[0], 1)),
                ],
                axis=-1,
            )
            core_hs = tf.concat(
                [
                    core,
                    tf.zeros((tf.shape(core)[0], 1)),
                    tf.ones((tf.shape(core)[0], 1)),
                ],
                axis=-1,
            )
            out_ls = (
                self.two_body_nn(core_ls, ligs, training=training)
                + self.three_body_nn_cis(core_ls, ligs, training=training)
                + self.three_body_nn_trans(core_ls, ligs, training=training)
            )
            out_hs = (
                self.two_body_nn(core_hs, ligs, training=training)
                + self.three_body_nn_cis(core_hs, ligs, training=training)
                + self.three_body_nn_trans(core_hs, ligs, training=training)
            )
            return self.output_norm(tf.concat([out_ls, out_hs], axis=-1), invert=True)
        return self.output_norm(
            self.two_body_nn(core, ligs, training=training)
            + self.three_body_nn_cis(core, ligs, training=training)
            + self.three_body_nn_trans(core, ligs, training=training),
            invert=True,
        )


class TwoBodyModel_combi(tf.keras.Model):
    def __init__(
        self,
        two_body_units=[16, 8],
        l2=0.01,
        dropout_rate=0.0,
        racs_norm=None,
        output_norms=None,
    ):
        super(TwoBodyModel, self).__init__()

        self.racs_norm = racs_norm
        self.output_norms = output_norms

        self.two_body_nns = {
            key: TwoBodyLayer(
                mlp_kw=dict(
                    hidden_units=two_body_units,
                    name=f"two-body-{key}",
                    kernel_regularizer=tf.keras.regularizers.L2(l2),
                    dropout_rate=dropout_rate,
                )
            )
            for key in ["sse", "homo", "gap"]
        }

    @tf.function
    def call(self, inputs, training=None):
        core = inputs["core"]
        ligs = self.racs_norm(inputs["ligands"])

        outputs = {}
        outputs["sse"] = self.output_norms["sse"](
            self.two_body_nns["sse"](core, ligs, training=training), invert=True
        )

        # Append one hot (2d) of spin state: LS=(1, 0), H=(0, 1)
        core_ls = tf.concat(
            [core, tf.ones((tf.shape(core)[0], 1)), tf.zeros((tf.shape(core)[0], 1))],
            axis=-1,
        )
        core_hs = tf.concat(
            [core, tf.zeros((tf.shape(core)[0], 1)), tf.ones((tf.shape(core)[0], 1))],
            axis=-1,
        )

        outputs["homo"] = self.output_norms["homo"](
            tf.concat(
                [
                    self.two_body_nns["homo"](core_ls, ligs, training=training),
                    self.two_body_nns["homo"](core_hs, ligs, training=training),
                ],
                axis=-1,
            ),
            invert=True,
        )

        outputs["gap"] = self.output_norms["gap"](
            tf.concat(
                [
                    self.two_body_nns["gap"](core_ls, ligs, training=training),
                    self.two_body_nns["gap"](core_hs, ligs, training=training),
                ],
                axis=-1,
            ),
            invert=True,
        )

        return outputs


class ThreeBodyModel_old(tf.keras.Model):
    cis_pairs = [
        (0, 1),
        (0, 3),
        (0, 4),
        (0, 5),
        (1, 2),
        (1, 4),
        (1, 5),
        (2, 3),
        (2, 4),
        (2, 5),
        (3, 4),
        (3, 5),
    ]
    trans_pairs = [(0, 2), (1, 3), (4, 5)]

    def __init__(
        self,
        two_body_units=[16, 8],
        three_body_units=[16, 8],
        l2=0.01,
        dropout_rate=0.0,
    ):
        super(ThreeBodyModel, self).__init__()

        self.two_body_nns = {
            key: MLP(
                hidden_units=two_body_units,
                name=f"two-body-{key}",
                kernel_regularizer=tf.keras.regularizers.L2(l2),
                dropout_rate=dropout_rate,
            )
            for key in ["sse", "homo", "gap"]
        }
        self.three_body_cis_nns = {
            key: MLP(
                hidden_units=three_body_units,
                name=f"three-body-cis-{key}",
                kernel_regularizer=tf.keras.regularizers.L2(l2),
                dropout_rate=dropout_rate,
            )
            for key in ["sse", "homo", "gap"]
        }
        self.three_body_trans_nns = {
            key: MLP(
                hidden_units=three_body_units,
                name=f"three-body-trans-{key}",
                kernel_regularizer=tf.keras.regularizers.L2(l2),
                dropout_rate=dropout_rate,
            )
            for key in ["sse", "homo", "gap"]
        }

    @tf.function
    def two_body(self, core, ligs, key):
        inp = tf.concat([tf.repeat(core[:, tf.newaxis, :], 6, axis=1), ligs], axis=-1)
        return self.two_body_nns[key](inp)

    @tf.function
    def three_body_cis(self, core, ligs, key):
        output = []
        for i, j in self.cis_pairs:
            inp1 = tf.concat([core, ligs[:, i, :], ligs[:, j, :]], axis=-1)
            inp2 = tf.concat([core, ligs[:, j, :], ligs[:, i, :]], axis=-1)
            res = 0.5 * (
                self.three_body_cis_nns[key](inp1) + self.three_body_cis_nns[key](inp2)
            )
            cond = tf.reduce_all(
                tf.math.abs(inp1 - inp2) < 1e-8, axis=-1, keepdims=True
            )
            output.append(tf.where(cond, tf.zeros_like(res), res))
        return tf.stack(output, axis=1)

    @tf.function
    def three_body_trans(self, core, ligs, key):
        output = []
        for i, j in self.trans_pairs:
            inp1 = tf.concat([core, ligs[:, i, :], ligs[:, j, :]], axis=-1)
            inp2 = tf.concat([core, ligs[:, j, :], ligs[:, i, :]], axis=-1)
            res = 0.5 * (
                self.three_body_trans_nns[key](inp1)
                + self.three_body_trans_nns[key](inp2)
            )
            cond = tf.reduce_all(
                tf.math.abs(inp1 - inp2) < 1e-8, axis=-1, keepdims=True
            )
            output.append(tf.where(cond, tf.zeros_like(res), res))
        return tf.stack(output, axis=1)

    @tf.function
    def call(self, inputs):
        core = inputs["core"]
        ligs = inputs["ligands"]

        outputs = {}
        outputs["sse"] = (
            tf.reduce_sum(self.two_body(core, ligs, "sse"), axis=-2)
            + tf.reduce_sum(self.three_body_cis(core, ligs, "sse"), axis=-2)
            + tf.reduce_sum(self.three_body_trans(core, ligs, "sse"), axis=-2)
        )

        # Append one hot (2d) of spin state: LS=(1, 0), H=(0, 1)
        core_ls = tf.concat(
            [core, tf.ones((tf.shape(core)[0], 1)), tf.zeros((tf.shape(core)[0], 1))],
            axis=-1,
        )
        core_hs = tf.concat(
            [core, tf.zeros((tf.shape(core)[0], 1)), tf.ones((tf.shape(core)[0], 1))],
            axis=-1,
        )

        outputs["homo_ls"] = (
            tf.reduce_sum(self.two_body(core_ls, ligs, "homo"), axis=-2)
            + tf.reduce_sum(self.three_body_cis(core_ls, ligs, "homo"), axis=-2)
            + tf.reduce_sum(self.three_body_trans(core_ls, ligs, "homo"), axis=-2)
        )
        outputs["homo_hs"] = (
            tf.reduce_sum(self.two_body(core_hs, ligs, "homo"), axis=-2)
            + tf.reduce_sum(self.three_body_cis(core_hs, ligs, "homo"), axis=-2)
            + tf.reduce_sum(self.three_body_trans(core_hs, ligs, "homo"), axis=-2)
        )

        outputs["gap_ls"] = (
            tf.reduce_sum(self.two_body(core_ls, ligs, "gap"), axis=-2)
            + tf.reduce_sum(self.three_body_cis(core_ls, ligs, "gap"), axis=-2)
            + tf.reduce_sum(self.three_body_trans(core_ls, ligs, "gap"), axis=-2)
        )
        outputs["gap_hs"] = (
            tf.reduce_sum(self.two_body(core_hs, ligs, "gap"), axis=-2)
            + tf.reduce_sum(self.three_body_cis(core_hs, ligs, "gap"), axis=-2)
            + tf.reduce_sum(self.three_body_trans(core_hs, ligs, "gap"), axis=-2)
        )

        return outputs
