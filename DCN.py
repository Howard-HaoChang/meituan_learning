import tensorflow as tf

def DCN(feature_dim, field_dim, embedding_dim, layer_num):
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
    embeddings = tf.reshape(embeddings, [-1, field_dim*embedding_dim])

    # Deep Crossing层
    with tf.variable_scope('dcn'):
        x_0 = embeddings
        x = embeddings
        for i in range(layer_num):
            x_i = x
            xx = tf.einsum('bi,bj->bij', x_0, x)
            x = tf.layers.dense(xx, 1, use_bias=True, activation=None)
            x = x + x_i

    # dense层
    out = tf.layers.dense(x, 1)

    # 损失函数
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=out, labels=label))

    # 优化器
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

    return feat_index, feat_value, label, loss, optimizer