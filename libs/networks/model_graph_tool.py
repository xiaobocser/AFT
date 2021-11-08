from __future__ import print_function
import tensorflow as tf
import numpy as np
import sys
import six
import math
from libs.help_utils.metrics import *
from libs.networks.model_tools import *
from libs.configs import cfgs


def add_embd_feature(feature_info_list, embedding_dim, index_num, feature_max_len, feature_name='default', input_type=tf.int32, embd_matrix_index=-1):
    if embd_matrix_index == -1:
        if False and feature_name.startswith('item_'):
            W = np.zeros(shape=(index_num, embedding_dim), dtype='float32')
            filename = 'graphwalk.emb'
            for line in open(filename):
                line_ele = line.rstrip().split('\t')
                i = int(line_ele[0])
                for j, val in enumerate(line_ele[1].split()):
                    W[i][j] = float(val)
            embd_matrix_ = tf.Variable(W, name=feature_name+'_matrix')
        else:
            embd_matrix_ = tf.get_variable(name=feature_name+'_matrix', initializer=tf.random_uniform_initializer(minval=-0.01, maxval=0.01), shape=[index_num, embedding_dim])
    else:
        embd_matrix_ = feature_info_list[embd_matrix_index]['embd_matrix']

    feature_info_list.append({'embedding_dim':embedding_dim, 'index_num':index_num, 'feature_max_len':feature_max_len, 'feature_name':feature_name, 
        'feature_input': tf.placeholder(input_type, [None, feature_max_len], name=feature_name+'_input'), 'embd_matrix': embd_matrix_ , 'type': 'embd'})


def add_feature(feature_info_list, feature_len, feature_name, input_type=tf.float32, feature_type='const', isnorm=False):
    tmp_f = tf.placeholder(input_type, [None, feature_len], name=feature_name+'_input')
    if isnorm:
        tmp_f = tf.nn.l2_normalize(tmp_f, dim=1)
    feature_info_list.append({'feature_input': tmp_f, 'type': feature_type, 'feature_name': feature_name, 'feature_len': feature_len})


def feature_init(hidden_size=64, max_term_num = 50, domain_cnt = 20, vocab_size = 17771 + 1):
    feature_info_list = []    

    embd_dim, embd_num, embd_length = hidden_size, domain_cnt + 1, 1
    add_embd_feature(feature_info_list, embd_dim, embd_num, embd_length, feature_name='channel_0')
    embd_dim, embd_num, embd_length = hidden_size, vocab_size, max_term_num
    add_embd_feature(feature_info_list, embd_dim, embd_num, embd_length, feature_name='item_0')

    embd_dim, embd_num, embd_length = hidden_size, domain_cnt+1, 1
    add_embd_feature(feature_info_list, embd_dim, embd_num, embd_length, feature_name='channel_1', embd_matrix_index=0)
    embd_dim, embd_num, embd_length = hidden_size, vocab_size, max_term_num
    add_embd_feature(feature_info_list, embd_dim, embd_num, embd_length, feature_name='item_1', embd_matrix_index=1)

    embd_dim, embd_num, embd_length = hidden_size, domain_cnt+1, 1
    add_embd_feature(feature_info_list, embd_dim, embd_num, embd_length, feature_name='channel_2', embd_matrix_index=0)
    embd_dim, embd_num, embd_length = hidden_size, vocab_size, max_term_num
    add_embd_feature(feature_info_list, embd_dim, embd_num, embd_length, feature_name='item_2', embd_matrix_index=1)

    embd_dim, embd_num, embd_length = hidden_size, domain_cnt+1, 1
    add_embd_feature(feature_info_list, embd_dim, embd_num, embd_length, feature_name='channel_3', embd_matrix_index=0)
    embd_dim, embd_num, embd_length = hidden_size, vocab_size, max_term_num
    add_embd_feature(feature_info_list, embd_dim, embd_num, embd_length, feature_name='item_3', embd_matrix_index=1)

    embd_dim, embd_num, embd_length = hidden_size, domain_cnt+1, 1
    add_embd_feature(feature_info_list, embd_dim, embd_num, embd_length, feature_name='channel_4', embd_matrix_index=0)
    embd_dim, embd_num, embd_length = hidden_size, vocab_size, max_term_num
    add_embd_feature(feature_info_list, embd_dim, embd_num, embd_length, feature_name='item_4', embd_matrix_index=1)

    embd_dim, embd_num, embd_length = hidden_size, domain_cnt+1, 1
    add_embd_feature(feature_info_list, embd_dim, embd_num, embd_length, feature_name='channel_5', embd_matrix_index=0)
    embd_dim, embd_num, embd_length = hidden_size, vocab_size, max_term_num
    add_embd_feature(feature_info_list, embd_dim, embd_num, embd_length, feature_name='item_5', embd_matrix_index=1)

    embd_dim, embd_num, embd_length = hidden_size, domain_cnt+1, 1
    add_embd_feature(feature_info_list, embd_dim, embd_num, embd_length, feature_name='channel_6', embd_matrix_index=0)
    embd_dim, embd_num, embd_length = hidden_size, vocab_size, max_term_num
    add_embd_feature(feature_info_list, embd_dim, embd_num, embd_length, feature_name='item_6', embd_matrix_index=1)

    embd_dim, embd_num, embd_length = hidden_size, domain_cnt+1, 1
    add_embd_feature(feature_info_list, embd_dim, embd_num, embd_length, feature_name='channel_7', embd_matrix_index=0)
    embd_dim, embd_num, embd_length = hidden_size, vocab_size, max_term_num
    add_embd_feature(feature_info_list, embd_dim, embd_num, embd_length, feature_name='item_7', embd_matrix_index=1)

    embd_dim, embd_num, embd_length = hidden_size, domain_cnt+1, 1
    add_embd_feature(feature_info_list, embd_dim, embd_num, embd_length, feature_name='channel_8', embd_matrix_index=0)
    embd_dim, embd_num, embd_length = hidden_size, vocab_size, max_term_num
    add_embd_feature(feature_info_list, embd_dim, embd_num, embd_length, feature_name='item_8', embd_matrix_index=1)

    embd_dim, embd_num, embd_length = hidden_size, domain_cnt+1, 1
    add_embd_feature(feature_info_list, embd_dim, embd_num, embd_length, feature_name='channel_9', embd_matrix_index=0)
    embd_dim, embd_num, embd_length = hidden_size, vocab_size, max_term_num
    add_embd_feature(feature_info_list, embd_dim, embd_num, embd_length, feature_name='item_9', embd_matrix_index=1)

    embd_dim, embd_num, embd_length = hidden_size, domain_cnt+1, 1
    add_embd_feature(feature_info_list, embd_dim, embd_num, embd_length, feature_name='channel_10', embd_matrix_index=0)
    embd_dim, embd_num, embd_length = hidden_size, vocab_size, max_term_num
    add_embd_feature(feature_info_list, embd_dim, embd_num, embd_length, feature_name='item_10', embd_matrix_index=1)

    embd_dim, embd_num, embd_length = hidden_size, domain_cnt+1, 1
    add_embd_feature(feature_info_list, embd_dim, embd_num, embd_length, feature_name='channel_11', embd_matrix_index=0)
    embd_dim, embd_num, embd_length = hidden_size, vocab_size, max_term_num
    add_embd_feature(feature_info_list, embd_dim, embd_num, embd_length, feature_name='item_11', embd_matrix_index=1)

    embd_dim, embd_num, embd_length = hidden_size, domain_cnt+1, 1
    add_embd_feature(feature_info_list, embd_dim, embd_num, embd_length, feature_name='channel_12', embd_matrix_index=0)
    embd_dim, embd_num, embd_length = hidden_size, vocab_size, max_term_num
    add_embd_feature(feature_info_list, embd_dim, embd_num, embd_length, feature_name='item_12', embd_matrix_index=1)

    embd_dim, embd_num, embd_length = hidden_size, domain_cnt+1, 1
    add_embd_feature(feature_info_list, embd_dim, embd_num, embd_length, feature_name='channel_13', embd_matrix_index=0)
    embd_dim, embd_num, embd_length = hidden_size, vocab_size, max_term_num
    add_embd_feature(feature_info_list, embd_dim, embd_num, embd_length, feature_name='item_13', embd_matrix_index=1)

    embd_dim, embd_num, embd_length = hidden_size, domain_cnt+1, 1
    add_embd_feature(feature_info_list, embd_dim, embd_num, embd_length, feature_name='channel_14', embd_matrix_index=0)
    embd_dim, embd_num, embd_length = hidden_size, vocab_size, max_term_num
    add_embd_feature(feature_info_list, embd_dim, embd_num, embd_length, feature_name='item_14', embd_matrix_index=1)

    embd_dim, embd_num, embd_length = hidden_size, domain_cnt+1, 1
    add_embd_feature(feature_info_list, embd_dim, embd_num, embd_length, feature_name='channel_15', embd_matrix_index=0)
    embd_dim, embd_num, embd_length = hidden_size, vocab_size, max_term_num
    add_embd_feature(feature_info_list, embd_dim, embd_num, embd_length, feature_name='item_15', embd_matrix_index=1)

    embd_dim, embd_num, embd_length = hidden_size, domain_cnt+1, 1
    add_embd_feature(feature_info_list, embd_dim, embd_num, embd_length, feature_name='channel_16', embd_matrix_index=0)
    embd_dim, embd_num, embd_length = hidden_size, vocab_size, max_term_num
    add_embd_feature(feature_info_list, embd_dim, embd_num, embd_length, feature_name='item_16', embd_matrix_index=1)

    embd_dim, embd_num, embd_length = hidden_size, domain_cnt+1, 1
    add_embd_feature(feature_info_list, embd_dim, embd_num, embd_length, feature_name='channel_17', embd_matrix_index=0)
    embd_dim, embd_num, embd_length = hidden_size, vocab_size, max_term_num
    add_embd_feature(feature_info_list, embd_dim, embd_num, embd_length, feature_name='item_17', embd_matrix_index=1)

    embd_dim, embd_num, embd_length = hidden_size, domain_cnt+1, 1
    add_embd_feature(feature_info_list, embd_dim, embd_num, embd_length, feature_name='channel_18', embd_matrix_index=0)
    embd_dim, embd_num, embd_length = hidden_size, vocab_size, max_term_num
    add_embd_feature(feature_info_list, embd_dim, embd_num, embd_length, feature_name='item_18', embd_matrix_index=1)

    embd_dim, embd_num, embd_length = hidden_size, domain_cnt+1, 1
    add_embd_feature(feature_info_list, embd_dim, embd_num, embd_length, feature_name='channel_19', embd_matrix_index=0)
    embd_dim, embd_num, embd_length = hidden_size, vocab_size, max_term_num
    add_embd_feature(feature_info_list, embd_dim, embd_num, embd_length, feature_name='item_19', embd_matrix_index=1)


    add_feature(feature_info_list, 2, 'f_len', feature_type='f_len')

    return feature_info_list


def feature_data_check(feature_info_list, res):
    for feature in feature_info_list:
        res.append(feature['feature_input'])


def feature_collect(pooling_list, feature_info_list, feature_type='const'):
    for feature in feature_info_list:
        if feature['type'] != feature_type: continue
        pooling_list.append(feature['feature_input'])


def tag_embd_lookup(finput, feature_info_list, isnorm=False, embd_matrix_=None, multi=False):
    if embd_matrix_ is not None:
        embed = tf.nn.embedding_lookup(embd_matrix_, finput)
    else:
        embed = tf.nn.embedding_lookup(feature_info_list[5]['embd_matrix'], finput)
    if not multi:
        embed = tf.reduce_sum(embed, axis=1)
    if isnorm:
        embed = tf.nn.l2_normalize(embed, dim=1)
    return embed

def local_embd_lookup(finput, feature_info_list, isnorm=False, embd_matrix_=None):
    if embd_matrix_ is not None:
        embed = tf.nn.embedding_lookup(embd_matrix_, finput)
    else:
        embed = tf.nn.embedding_lookup(feature_info_list[0]['embd_matrix'], finput)
    embed = tf.reduce_sum(embed, axis=1)
    if isnorm:
        embed = tf.nn.l2_normalize(embed, dim=1)
    return embed


def tag_embd_mp(feature_info_list, isnorm=False, embd_matrix_=None, range_cnt=-1, tagids=None):
    if range_cnt == -1:
        if tagids is None:
            mp_tag = np.array(range(0, cfgs.vocab_size))
        else:
            mp_tag = tagids
    else:
        mp_tag = np.array(range(0, range_cnt))

    if embd_matrix_ is not None:
        embed = tf.nn.embedding_lookup(embd_matrix_, mp_tag)
    else:
        embed = tf.nn.embedding_lookup(feature_info_list[5]['embd_matrix'], mp_tag)
    if isnorm:
        embed = tf.nn.l2_normalize(embed, dim=1)
    return embed

def local_embd_mp2(feature_info_list, isnorm=False, embd_matrix_=None):
    mp_tag = np.array([2 for i in range(0, 17770)])
    if embd_matrix_ is not None:
        embed = tf.nn.embedding_lookup(embd_matrix_, mp_tag)
    else:
        embed = tf.nn.embedding_lookup(feature_info_list[1]['embd_matrix'], mp_tag)
    if isnorm:
        embed = tf.nn.l2_normalize(embed, dim=1)
    return embed

def local_embd_mp17(feature_info_list, isnorm=False, embd_matrix_=None):
    mp_tag = np.array([17 for i in range(0, 17770)])
    if embd_matrix_ is not None:
        embed = tf.nn.embedding_lookup(embd_matrix_, mp_tag)
    else:
        embed = tf.nn.embedding_lookup(feature_info_list[1]['embd_matrix'], mp_tag)
    if isnorm:
        embed = tf.nn.l2_normalize(embed, dim=1)
    return embed

def feature_embd_lookup(pooling_list, it_list, domain_item_list, feature_info_list, max_term_num, summaries_list=None, is_trainsum_flag=False, isnorm=True):
    index = 0
    for feature in feature_info_list:
        if feature['type'] != 'embd': continue
        # [B,1/max_term]
        it_list.append(feature['feature_input'])
        # [B,1/max_term,E]
        embed = tf.nn.embedding_lookup(feature['embd_matrix'], feature['feature_input'])
        pooling_list.append(embed)

        domain_item_rate = tf.reduce_sum(tf.to_float(tf.abs(feature['feature_input']) > 0), -1, keep_dims=True) / float(max_term_num)
        domain_item_list.append(domain_item_rate)

        index += 1
        if is_trainsum_flag and summaries_list is not None:
            summaries_list.append(tf.summary.histogram(feature['feature_name']+'_embeding', feature['embd_matrix']))
            summaries_list.append(tf.summary.histogram(feature['feature_name']+'_pooling', embed))

#field_embd = tf.concat([embd_list[0], embd_list[2], embd_list[4], embd_list[6], embd_list[8], embd_list[10], embd_list[12]] , 1)
# B * D * E
def cnn_layer(input_embd, input_shape, weight_shape, strides_shape, output_shape, cnn_name, padname='SAME', summaries_list=None, is_trainsum_flag=False):
    input_x = tf.reshape(input_embd, input_shape)
    weight = tf.Variable(tf.truncated_normal(shape=weight_shape, stddev=0.1), name=cnn_name+'cnnW')
    #cnn_layer = tf.nn.conv2d(input_x, weight, strides=[1, 7, 3, 1], padding = 'SAME')
    cnn_x = tf.nn.conv2d(input_x, weight, strides=strides_shape, padding = padname)
    output_x = tf.reshape(cnn_x, output_shape)
    cnn_layer = tf.reduce_sum(output_x, 1)
    
    #user_attention_layer = tf.reshape(user_embedding * tf.expand_dims(alphas_u, -1), [self.batch_size, -1])
    if summaries_list is not None and is_trainsum_flag:
        summaries_list.append(tf.summary.histogram(cnn_name+'cnnW', weight))
    
    return cnn_layer

def fm_layer(input_embd, input_size, output_size, hidden_size, layer_name, summaries_list=None, is_trainsum_flag=False, parameter=None, func=tf.nn.leaky_relu, is_training=False):
    #W = tf.get_variable(layer_name+'_W', shape=[input_size, output_size], initializer=tf.contrib.layers.xavier_initializer())
    V = tf.get_variable(layer_name+'_V', shape=[input_size, hidden_size], initializer=tf.contrib.layers.variance_scaling_initializer())
    W = tf.get_variable(layer_name+'_W', shape=[input_size, output_size], initializer=tf.contrib.layers.variance_scaling_initializer())
    b = tf.get_variable(layer_name+'_b', shape=[output_size], initializer=tf.constant_initializer(0.0))
    
    #input_embd = tf.layers.batch_normalization(input_embd, training=is_training, name=layer_name+'_bn')
#    layer = func(tf.matmul(input_embd, W) + b, name=layer_name+'_layer')

    fm = tf.square(tf.matmul(input_embd, V)) - tf.matmul(tf.multiply(input_embd, input_embd), tf.multiply(V, V)) 
    liner = tf.matmul(input_embd, W) + b
    fm_all = tf.concat([liner, fm], 1)

    bn_flag = False
    if bn_flag:
        fm_all = tf.layers.batch_normalization(fm_all, training=is_training, name=layer_name+'_bn')

    layer = func(fm_all, name=layer_name+'_layer')

    if summaries_list is not None and is_trainsum_flag:
        summaries_list.append(tf.summary.histogram(layer_name+'layer', layer))
        summaries_list.append(tf.summary.histogram(layer_name+'_W', W))
        summaries_list.append(tf.summary.histogram(layer_name+'_b', b))
    if parameter is not None:
        parameter[layer_name+'_W'] = W
        parameter[layer_name+'_b'] = b

    return layer

def dense_layer(input_embd, input_size, output_size, layer_name, summaries_list=None, is_trainsum_flag=False, parameter=None, func=tf.nn.leaky_relu, is_training=False):
    #W = tf.get_variable(layer_name+'_W', shape=[input_size, output_size], initializer=tf.contrib.layers.xavier_initializer())
    W = tf.get_variable(layer_name+'_W', shape=[input_size, output_size], initializer=tf.contrib.layers.variance_scaling_initializer())
    b = tf.get_variable(layer_name+'_b', shape=[output_size], initializer=tf.constant_initializer(0.0))
    
    bn_flag = False
    if not bn_flag:
        layer = func(tf.matmul(input_embd, W) + b, name=layer_name+'_layer')
    else:
        linear_output = tf.matmul(input_embd, W)
        batch_normalized_output = tf.layers.batch_normalization(linear_output, training=is_training, name=layer_name+'_bn')
        layer = func(batch_normalized_output, name=layer_name+'_layer')

    if summaries_list is not None and is_trainsum_flag:
        summaries_list.append(tf.summary.histogram(layer_name+'layer', layer))
        summaries_list.append(tf.summary.histogram(layer_name+'_W', W))
        summaries_list.append(tf.summary.histogram(layer_name+'_b', b))
    if parameter is not None:
        parameter[layer_name+'_W'] = W
        parameter[layer_name+'_b'] = b

    return layer

def optimizer_fuc(cross_entropy, global_step, learning_rate=1e-5, func_name='adam'):
    # neg sample op
    if func_name == 'gd':
        opt = tf.train.GradientDescentOptimizer(learning_rate)
    if func_name == 'adam':
        opt = tf.train.AdamOptimizer(learning_rate)
    if func_name == 'rmsprop':
        opt = tf.train.RMSPropOptimizer(learning_rate)

    return opt.minimize(cross_entropy, global_step)

def auc_calculate(targets_input, predict_input, input_name='default', summaries_list=None):
    streaming_auc = tf.metrics.auc(labels=targets_input, predictions=predict_input)
    auc_res = tf.reduce_mean(streaming_auc)
    if summaries_list is not None:
        summaries_list.append( tf.summary.scalar("auc_" + input_name, auc_res) )
    return auc_res

def recall_calculate(targets_input, predict_input, topk = 20, input_name='default', summaries_list=None):
    targets_input = tf.cast(targets_input, dtype=tf.int64)

    recall_res = tf.reduce_mean(tf.map_fn(lambda x: recall_fun(
        predictions=tf.sigmoid(x[0]), labels=x[1], topk=topk), (predict_input, targets_input), dtype=tf.float64))

    return recall_res

def get_input_batches(file_list, batch_size):
    """
    Fetch input data from a queue
    """
    filename_queue = tf.train.string_input_producer(file_list, shuffle=True, num_epochs=None)
    reader = tf.TextLineReader()
    _, line = reader.read(filename_queue)

    min_after_dequeue = 1000
    capacity = min_after_dequeue + 3 * batch_size
    return tf.train.shuffle_batch(
                [line],
                batch_size=batch_size*3,
                capacity=capacity,
                min_after_dequeue=min_after_dequeue)