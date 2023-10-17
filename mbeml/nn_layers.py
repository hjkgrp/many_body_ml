import tensorflow as tf


class CustomNormalization(tf.keras.layers.Layer):
    def __init__(self, mean, std, **kwargs):
        super(CustomNormalization, self).__init__(**kwargs)
        self.mean = tf.Variable(
            initial_value=tf.cast(mean, tf.keras.backend.floatx()),
            trainable=False,
            name="mean",
        )
        self.std = tf.Variable(
            initial_value=tf.cast(std, tf.keras.backend.floatx()),
            trainable=False,
            name="std",
        )

    @tf.function
    def call(self, inputs, invert=False):
        if invert:
            return inputs * tf.maximum(self.std, tf.keras.backend.epsilon()) + self.mean
        return (inputs - self.mean) / tf.maximum(self.std, tf.keras.backend.epsilon())


class AddSpinEncoding(tf.keras.layers.Layer):
    @tf.function
    def call(self, core_inp):
        # Reference for a (batch_size x 1) tensor
        batch_shape = core_inp[:, :1]
        zeros_like_core = tf.zeros_like(batch_shape)
        ones_like_core = tf.ones_like(batch_shape)
        core_ls = tf.concat(
            [core_inp, ones_like_core, zeros_like_core], axis=-1, name="core_input_ls"
        )
        core_hs = tf.concat(
            [core_inp, zeros_like_core, ones_like_core], axis=-1, name="core_input_hs"
        )
        return core_ls, core_hs


class TwoBodyPrep(tf.keras.layers.Layer):
    @tf.function
    def call(self, inputs):
        core, ligs = inputs
        return tf.concat([tf.repeat(core[:, tf.newaxis, :], 6, axis=1), ligs], axis=-1)


class ThreeBodyPrep(tf.keras.layers.Layer):
    def __init__(self, pairs, **kwargs):
        super(ThreeBodyPrep, self).__init__(**kwargs)
        self.pairs = pairs

    @tf.function
    def call(self, inputs):
        core, ligs = inputs
        output = []
        for i, j in self.pairs:
            inp1 = tf.concat([core, ligs[:, i, :], ligs[:, j, :]], axis=-1)
            inp2 = tf.concat([core, ligs[:, j, :], ligs[:, i, :]], axis=-1)
            output.append(tf.stack([inp1, inp2], axis=1))
        return tf.stack(output, axis=1)


class ThreeBodyPrepFeatureSym(tf.keras.layers.Layer):
    def __init__(self, pairs, **kwargs):
        super(ThreeBodyPrepFeatureSym, self).__init__(**kwargs)
        self.pairs = pairs

    @tf.function
    def call(self, inputs):
        core, ligs = inputs
        output = []
        for i, j in self.pairs:
            inp = tf.concat(
                [
                    core,
                    0.5 * (ligs[:, i, :] + ligs[:, j, :]),
                    0.5 * tf.abs(ligs[:, i, :] - ligs[:, j, :]),
                ],
                axis=-1,
            )
            output.append(inp)
        return tf.stack(output, axis=1)


class ThreeBodyMask(tf.keras.layers.Layer):
    def __init__(self, pairs, **kwargs):
        super(ThreeBodyMask, self).__init__(**kwargs)
        self.pairs = pairs

    @tf.function
    def call(self, inputs):
        core, ligs = inputs
        output = []
        for i, j in self.pairs:
            inp1 = tf.concat([core, ligs[:, i, :], ligs[:, j, :]], axis=-1)
            inp2 = tf.concat([core, ligs[:, j, :], ligs[:, i, :]], axis=-1)
            cond = tf.reduce_any(
                tf.math.abs(inp1 - inp2) > 1e-6, axis=-1, keepdims=True
            )
            output.append(tf.cast(cond, tf.keras.backend.floatx()))
        return tf.stack(output, axis=1)
