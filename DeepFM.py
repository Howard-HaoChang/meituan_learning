# coding=utf-8
import tensorflow as tf

def DeepFM(feature_dim, field_dim, embedding_dim, mlp_dim):
    # 输入层
    feat_index = tf.placeholder(tf.int32, shape=[None, None], name='feat_index')  # Batch * Feature
    feat_value = tf.placeholder(tf.int32, shape=[None, None], name='feat_value')  # Batch * Field
    shape_value = tf.shape(feat_value)
    range_vector = tf.tile(tf.reshape(tf.range(shape_value[1]), [1, shape_value[1]]), [shape_value[0], 1])
    feat_value = tf.add(feat_value, range_vector)
    label = tf.placeholder(tf.float32, shape=[None, 1], name='label')  # Batch * 1

    # Embedding层
    weights = {'feature_embeddings': tf.Variable(tf.random_normal([feature_dim, embedding_dim], 0.0, 0.01), name='feature_embeddings')}  # Feature * Embedding
    embeddings = tf.nn.embedding_lookup(weights['feature_embeddings'], feat_index)  # Batch * Field * Embedding

    # FM层
    # 一阶部分
    fm_linear = tf.layers.dense(tf.reduce_sum(embeddings, 2), 1)  # Batch * 1
    # 二阶部分
    fm_quadratic = 0.5 * tf.reduce_sum(
        tf.square(tf.reduce_sum(embeddings, 1)) - tf.reduce_sum(tf.square(embeddings), 1), 1, keep_dims=True
    )  # Batch * 1

    # Dense层
    l1 = tf.layers.dense(embeddings, 10, activation=tf.nn.relu)  # Batch * 10
    l2 = tf.layers.dense(l1, 1)  # Batch * 1

    # 输出层
    out = tf.add_n(fm_linear, fm_quadratic, l2)

    # 损失函数
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=out, labels=label))

    # 优化器
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

    return feat_index, feat_value, label, loss, optimizer