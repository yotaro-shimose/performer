from performer.linear_attention import LinearAttention
import tensorflow as tf


def test_softmax_kernel_approximation():
    B = 64
    L = 128
    F = 128
    M = 64
    stddev = 0.01
    layer = LinearAttention(M)
    q = tf.random.normal((B, L, F), stddev=stddev)
    k = tf.random.normal((B, L, F), stddev=stddev)
    layer._build_with_F(F)
    unnormalized = tf.matmul(layer._phi(q), layer._phi(k), transpose_b=True)
    performer = unnormalized / \
        tf.reduce_sum(unnormalized, axis=-1, keepdims=True)
    transformer = tf.math.softmax(
        tf.matmul(q, k, transpose_b=True) / tf.sqrt(float(F)), axis=-1)
    assert performer.shape == transformer.shape
    dif_kernel_approximation = performer - transformer

    x = tf.random.normal((B, L, F), stddev=stddev)
    y = tf.random.normal((B, L, F), stddev=stddev)
    x /= tf.reduce_sum(x, axis=-1, keepdims=True)
    y /= tf.reduce_sum(x, axis=-1, keepdims=True)
    dif_random_approximation = x - y

    assert tf.reduce_sum(tf.square(dif_kernel_approximation)) * \
        10000 < tf.reduce_sum(tf.square(dif_random_approximation))
