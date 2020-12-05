import tensorflow as tf
from performer.linear_attention import MultiHeadLinearAttention


def test_performer_output_shape():
    B = 10
    L = 1000
    F = 128
    M = 32
    stddev = 0.01

    d_value = 64
    d_key = 128
    n_heads = 8
    n_omega = 32

    q = tf.random.normal((B, L, F), stddev=stddev)
    k = tf.random.normal((B, L, F), stddev=stddev)
    v = tf.random.normal((B, L, F), stddev=stddev)
    layer = MultiHeadLinearAttention(
        d_value=d_value, d_key=d_key, n_heads=n_heads, n_omega=n_omega)
    performer = layer(q, k, v)
    assert performer.shape == (B, L, d_value)
