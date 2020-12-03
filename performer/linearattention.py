import tensorflow as tf


class LinearAttention(tf.keras.layers.Layer):
    def __init__(self, n_omega=64, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._built_with_F = False
        self.n_omega = n_omega

    def _build_with_F(self, d_feature):
        F = d_feature
        M = self.n_omega
        self.omega: tf.Variable = self.add_variable(
            shape=(M, F),
            trainable=False
        )
        self._built_with_F = True
        self.omega.assign(self._draw_omega(d_feature))

    def _draw_omega(self, d_feature: int):
        # F, F
        random_features = tf.random.normal((d_feature, d_feature))
        # F, F
        q, _ = tf.linalg.qr(random_features)
        q = q * tf.sqrt(float(d_feature))
        # M, F
        return q[:self.n_omega]

    def _phi(self, x: tf.Tensor):
        """

        Args:
            x (tf.Tensor): B, T, F
        """
        # B, T, 1
        h = tf.exp(-tf.reduce_sum(tf.square(x), axis=-1,
                                  keepdims=True)/2.) / tf.sqrt(2.)
        # B, T, M
        xw = tf.einsum("...ik,...jk->...ij", x, self.omega)
        # B, T, 2M
        unnormalized = tf.concat([tf.exp(xw), tf.exp(-xw)], axis=-1)
        normalizer = tf.sqrt(float(self.n_omega))
        return h * unnormalized / normalizer

    def call(self, query: tf.Tensor, value: tf.Tensor, key: tf.Tensor) -> tf.Tensor:
        """

        Args:
            query (tf.Tensor): B, T, F
            value (tf.Tensor): B, S, D
            key (tf.Tensor): B, S, F

        Returns:
            tf.Tensor: B, T, D
        """
        if not self._built_with_F:
            self._build_with_F(value.shape[-1])
        # B, T, M
        query = self._phi(query)
        # B, S, M
        key = self._phi(key)
        # debug
        return tf.matmul(query, key, transpose_b=True)
        # ones_shape = value.shape
        # ones_shape[-1] = 1
        # # B, S, D + 1
        # value = tf.concat(
        #     [value, tf.ones(ones_shape, dtype=tf.float32)], axis=-1)
        # # B, M, D + 1
        # keyvalue = tf.einsum("...ki,...kj->ij", key, value)
        # # B, T, D + 1
        # qkv = tf.einsum("...ik,...kj->ij", query, keyvalue)
        # # B, T, 1
        # normalizer = tf.expand_dims(qkv[:, :, -1], -1)
        # # B, T, D
        # qkv = qkv[:, :, :-1]

        # return qkv / normalizer
