# coding=utf-8
import tensorflow as tf

def PNN(feature_dim, field_dim, embedding_dim, mlp_dim):
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

    # Product层toDense层
    # 线性部分
    weights['embeddings_to_product_linear'] = tf.Variable(tf.random_normal([field_dim, embedding_dim, mlp_dim], 0.0, 0.01), name='embeddings_to_product_linear')
    lz = tf.tensordot(embeddings, weights['embeddings_to_product_linear'], [[1, 2], [0, 1]])  # Batch * MLP
    # 非线性部分
    theta = tf.Variable(tf.random_normal([field_dim, embedding_dim, mlp_dim], 0.0, 0.01), name='theta')
    lp = tf.norm(tf.reduce_sum(tf.expand_dims(embeddings, -1) * tf.expand_dims(theta, 0), axis=1), axis=1)  # Batch * MLP
    # toDense层
    weights['bias'] = tf.Variable(tf.constant(0.01, shape=[10]), name='bias')
    l1 = tf.nn.relu(tf.add(lz+lp+weights['bias']))

    # 全连接层
    l2 = tf.layers.dense(l1, 10, activation=tf.nn.relu)  # Batch * 10
    out = tf.layers.dense(l2, 1)  # Batch * 1

    # 损失函数
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=out, labels=label))

    # 优化器
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

    return feat_index, feat_value, label, loss, optimizer