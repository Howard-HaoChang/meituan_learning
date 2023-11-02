# coding=utf-8
import tensorflow as tf

def AFM(feature_dim, field_dim, embedding_dim, attention_dim):
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

    # Pair-Wise Interaction层
    interaction = []
    for i in range(field_dim):
        for j in range(i+1,field_dim):
            interaction.append(tf.multiply(embeddings[:,i,:], embeddings[:,j,:]))
    interaction = tf.stack(interaction)
    interaction = tf.transpose(interaction, perm = [1,0,2])  # Batch * M(M-1)/2 * Embedding

    # Attention-Based pooling层
    att_1 = tf.layers.dense(interaction, attention_dim, activation = tf.nn.relu)
    att_2 = tf.layers.dense(att_1, 1, use_bias = False)
    att_3 = tf.nn.softmax(tf.squeeze(att_2, -1))  # Batch * M(M-1)/2
    l_pooling = tf.reduce_sum(tf.multiply(tf.expand_dims(att_3, axis = -1), embeddings), axis = 1)  # Batch * Embedding

    # Dense层
    dense1 = tf.layers.dense(tf.reduce_sum(embeddings, 2), 1)
    dense2 = tf.layers.dense(l_pooling)
    out = tf.add(dense1, dense2)  # Batch * 1

    # 损失函数
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=out, labels=label))

    # 优化器
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

    return feat_index, feat_value, label, loss, optimizer