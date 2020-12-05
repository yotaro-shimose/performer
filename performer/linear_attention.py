import tensorflow as tf


class LinearAttention(tf.keras.layers.Layer):
    def __init__(self, n_omega=64, eps=1e-9, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._built_with_F = False
        self.n_omega = n_omega
        self.eps = eps

    def _build_with_F(self, d_feature):
        F = d_feature
        M = self.n_omega
        self.omega: tf.Variable = self.add_weight(
            shape=(M, F),
            trainable=False
        )
        self._built_with_F = True
        self.omega.assign(self._draw_omega(d_feature))

    def _draw_omega(self, d_feature: int):
        n_full_stack = self.n_omega // d_feature
        remainder = self.n_omega - n_full_stack * d_feature
        stack = tf.TensorArray(dtype=tf.float32, size=0,
                               dynamic_size=True, infer_shape=False)
        for i in tf.range(n_full_stack):
            # F, F
            random_features = tf.random.normal((d_feature, d_feature))
            # F, F
            q, _ = tf.linalg.qr(random_features)
            q = q * tf.sqrt(float(d_feature))
            stack.write(i, q)

        random_features = tf.random.normal((d_feature, d_feature))
        # F, F
        q, _ = tf.linalg.qr(random_features)
        q = q * tf.sqrt(float(d_feature))
        stack.write(n_full_stack, q[:remainder])
        # M, F
        return stack.concat()

    def _phi(self, x: tf.Tensor):
        """

        Args:
            x (tf.Tensor): B, T, F
        """
        # B, T, 1
        square_sum = tf.reduce_sum(tf.square(x), axis=-1, keepdims=True) / 2.
        # B, T, M
        xw = tf.einsum("...ik,...jk->...ij", x, self.omega)
        # B, T, 2M
        unnormalized = tf.concat(
            [xw, -xw], axis=-1) - square_sum
        unnormalized = tf.exp(unnormalized) + self.eps
        normalizer = tf.sqrt(2. * self.n_omega)
        normalized = unnormalized / normalizer
        return normalized

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
