from __future__ import print_function
import tensorflow as tf
import math
from tensorflow.contrib import layers
import numpy as np
from libs.configs import cfgs
from libs.networks.model_graph_tool import *


class ModelGraph():

    def __init__(self, test_file_list,
                 test_pos_num=10,
                 test_neg_num = 50,
                 is_trainsum_flag=True,
                 ffn_size=128,
                 vocab_size = 17772,
                 g_neg_cnt=60,
                 max_term_num=50,
                 num_attention_head=16,
                 num_hidden_layers=1,
                 hidden_size=64,
                 hidden_drop_prob=0.1,
                 attention_drop_prob=0.1,
                 initializer_range=0.02,
                 domain_cnt = 20,
                 cnn_size = 16
                 ):

        with tf.name_scope('init_param'):

            self.epoch = 5e4
            self.batch_size_scale = 1
            self.batch_size = 256 * self.batch_size_scale

            self.neg_label_count = 50
            self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
            self.dp_val = 0.5
            self.global_step = tf.Variable(0, trainable=False)
            self.is_training = tf.placeholder(tf.bool, name='is_training')

            self.lr = 0.005

            self.test_pos_num = test_pos_num

            train_summaries, test_summaries = [], []

            # auc
            self.label_input = tf.placeholder(tf.float32, [None, 1], name='label_input')
            self.labelid = tf.placeholder(tf.int32, [None, 1], name='labelid_input')
            self.targets_input = tf.placeholder(tf.bool, [None, 1], name="targets_input")

            # recall
            self.label_input_recall = tf.placeholder(tf.float32, [None, test_pos_num + test_neg_num], name='label_input_recall')
            self.labelid_recall = tf.placeholder(tf.int32, [None, test_pos_num + test_neg_num], name='labelid_input_recall')
            self.targets_input_recall = tf.placeholder(tf.bool, [None, test_pos_num + test_neg_num], name="targets_input_recall")

            self.label_input_gneg = tf.placeholder(tf.float32, [None, g_neg_cnt], name='label_input_gneg')
            self.labelid_gneg = tf.placeholder(tf.int32, [None, g_neg_cnt], name='labelid_input_gneg')
            self.targets_input_gneg = tf.placeholder(tf.bool, [None, g_neg_cnt], name="targets_input_gneg")

            self.parameter = {}

        with tf.name_scope('embedding_init'):
            self.feature_info_list = feature_init(hidden_size=hidden_size,max_term_num=max_term_num,domain_cnt=domain_cnt,vocab_size = vocab_size)
            self.labelLoid = tf.placeholder(tf.int32, [None, 1], name='labelLoid_input')
            self.tagids = tf.placeholder(tf.int32, [10000], name='tagids_input')

        embd_list, id_list, domain_item_rate_list, const_list, vec_list, flen_list = [], [], [], [],[],[]
        with tf.name_scope('feature_embedding'):
            feature_embd_lookup(embd_list, id_list, domain_item_rate_list, self.feature_info_list, max_term_num, train_summaries, is_trainsum_flag, isnorm=False)
            feature_collect(const_list, self.feature_info_list, 'const')
            feature_collect(flen_list, self.feature_info_list, 'f_len')

            #[B,E]
            tag_embd = tag_embd_lookup(self.labelid, self.feature_info_list)
            #[B,T,E]
            tag_embd_gneg = tag_embd_lookup(self.labelid_gneg, self.feature_info_list, multi=True)

            tag_embd_recall = tag_embd_lookup(self.labelid_recall, self.feature_info_list, multi=True)

            local_embd = local_embd_lookup(self.labelLoid, self.feature_info_list)

            const_len = sum(x['feature_len'] for index, x in enumerate(self.feature_info_list) if x['type'] == 'const')
            f_len = sum(x['feature_len'] for index, x in enumerate(self.feature_info_list) if x['type'] == 'f_len')

        with tf.name_scope('domain_encoder'):
            atten_head, head_size = num_attention_head, hidden_size

            #  -------------------------- domain feature process --------------------------------------
            field_embd_list = []
            domain_rate_list = []
            mask_list = []
            for i in range(domain_cnt):
                # embed_bool [B,1]
                embed_bool = tf.to_float(tf.abs(id_list[i * 2]) > 0)
                mask_list.append(embed_bool)
                field_embd = tf.multiply(embd_list[i * 2], tf.expand_dims(embed_bool,-1))

                field_embd_list.append(field_embd)
                domain_rate_list.append(tf.expand_dims(domain_item_rate_list[i * 2 + 1],1))
            # field_expand_layer [B, D, E]
            field_expand_layer = tf.multiply(tf.concat(field_embd_list, 1),tf.concat(domain_rate_list,1))

            field_attention_mask = create_attention_mask_from_input_mask(
                field_expand_layer, tf.concat(mask_list,axis=1))
            # field_atten_layer [B, T, N * H]
            field_encoder_layer = attention_layer(
                from_tensor=field_expand_layer,
                to_tensor=field_expand_layer,
                is_training=self.is_training,
                attention_mask=field_attention_mask,
                num_attention_heads=1,
                size_per_head=head_size,
                attention_probs_dropout_prob=attention_drop_prob,
                initializer_range=initializer_range,
                batch_size=self.batch_size,
                from_seq_length=len(embd_list) / 2,
                to_seq_length=len(embd_list) / 2,
                suffix="field_attention")

            field_encoder_layer = tf.layers.dense(
                field_encoder_layer,
                hidden_size,
                kernel_initializer=create_initializer(initializer_range))
            field_encoder_layer = dropout(field_encoder_layer, hidden_drop_prob, is_training=self.is_training)
            field_encoder_layer = layer_norm(field_encoder_layer + field_expand_layer)


        with tf.name_scope('item_encoder'):
            #  -------------------------- item feature process --------------------------------------
            # items_expand_layer [B, D, E]
            item_embd_list = []

            item_channel_mask_list = []
            for i in range(domain_cnt):
                item_input = id_list[i * 2 + 1]
                item_mask = tf.to_float(tf.abs(item_input) > 0)
                item_channel_mask_list.append(tf.to_float(tf.reduce_sum(item_mask, 1, keep_dims=True) > 0))

                item_embedding_output = embd_list[i * 2 + 1]
                item_attention_mask = create_attention_mask_from_input_mask(
                    item_embedding_output, item_mask)
                # `channel_layer`: [B, T, N * H]
                item_encoder = attention_layer(
                    from_tensor=item_embedding_output,
                    to_tensor=item_embedding_output,
                    is_training=self.is_training,
                    attention_mask=item_attention_mask,
                    num_attention_heads=1,
                    size_per_head=hidden_size,
                    attention_probs_dropout_prob=attention_drop_prob,
                    initializer_range=initializer_range,
                    batch_size=self.batch_size,
                    from_seq_length=max_term_num,
                    to_seq_length=max_term_num,
                    suffix="channel_attention_%s" % i)

                item_encoder = tf.layers.dense(
                    item_encoder,
                    hidden_size,
                    kernel_initializer=create_initializer(initializer_range))
                item_encoder = dropout(item_encoder, hidden_drop_prob, is_training=self.is_training)
                item_encoder = layer_norm(item_encoder + item_embedding_output)

                # [B,1,E]
                item_encoder_pool = tf.reduce_sum(item_encoder, axis=1,keep_dims=True)

                item_embd_list.append(item_encoder_pool)

            # items_encoder_layer [B,D,E]
            items_encoder_layer = tf.concat(item_embd_list, 1)


        with tf.name_scope('transformer_cross'):

            item_channel_mask = tf.concat(item_channel_mask_list, axis=1)
            item_channel_attention_mask = create_attention_mask_from_input_mask(
                items_encoder_layer, item_channel_mask)
            # item_atten_layer [B, D, E]
            items_transformer_layer = transformer_model(input_tensor=items_encoder_layer,
                                                    is_training=self.is_training,
                                                    attention_mask=item_channel_attention_mask,
                                                    hidden_size=hidden_size,
                                                    num_hidden_layers=num_hidden_layers,
                                                    num_attention_heads=num_attention_head,
                                                    intermediate_size=ffn_size,
                                                    hidden_dropout_prob=hidden_drop_prob,
                                                    attention_probs_dropout_prob=attention_drop_prob,
                                                    initializer_range=initializer_range,
                                                    do_return_all_layers=False,
                                                    attention_name="item_attention"
                                                    )

            field_transformer_layer = transformer_model(input_tensor=field_encoder_layer,
                                                    is_training=self.is_training,
                                                    attention_mask=field_attention_mask,
                                                    hidden_size=hidden_size,
                                                    num_hidden_layers=num_hidden_layers,
                                                    num_attention_heads=num_attention_head,
                                                    intermediate_size=ffn_size,
                                                    hidden_dropout_prob=hidden_drop_prob,
                                                    attention_probs_dropout_prob=attention_drop_prob,
                                                    initializer_range=initializer_range,
                                                    do_return_all_layers=False,
                                                    attention_name="field_attention"
                                                    )

        with tf.name_scope('conE'):
            # ----------------------------------   conE  ----------------------------------------------------
            # field_encoder_layer: [B,D,E]  items_encoder_layer: [B,D,E]
            conE_list = []
            for i in range(domain_cnt):
                conE_list.append(tf.slice(field_transformer_layer, [0, i, 0], [-1, 1, head_size]))
                conE_list.append(tf.slice(items_transformer_layer, [0, i, 0], [-1, 1, head_size]))
            # conE_tensor: [B,2 * D,E]
            conE_tensor = tf.concat(conE_list, 1)

            # [B,E*C]
            conE_cnn_layer = cnn_layer(conE_tensor, [-1, 2 * domain_cnt, head_size, 1], [2, 2, 1, cnn_size],
                                       [1, 1, 1, 1],
                                       [-1, 2 * domain_cnt, head_size * cnn_size], 'conE_cnn_g', 'SAME', train_summaries,
                                       is_trainsum_flag)

            conE_cnn_layer = tf.reshape(conE_cnn_layer, [-1, head_size, cnn_size])

            # conE_cnn_layer [B,C,E]
            conE_cnn_layer = tf.transpose(conE_cnn_layer, perm=[0, 2, 1])

            # target_onehot [B,D]
            target_onehot = tf.reduce_sum(tf.one_hot(self.labelLoid, domain_cnt), axis=1)

            # field_target_embd [B,1,E]
            field_target_embd = tf.reduce_sum(tf.multiply(field_encoder_layer, tf.expand_dims(target_onehot, -1)),
                                              axis=1, keep_dims=True)
            item_target_embd = tf.reduce_sum(tf.multiply(items_encoder_layer, tf.expand_dims(target_onehot, -1)),
                                             axis=1,keep_dims=True)

            conE_list = []
            for i in range(cnn_size):
                conE_list.append(tf.slice(conE_cnn_layer, [0, i, 0], [-1, 1, head_size]))
                conE_list.append(field_target_embd)
                conE_list.append(item_target_embd)
            conE_tensor = tf.concat(conE_list, 1)

            # [B,E*C]
            conE_cnn_layer_t = cnn_layer(conE_tensor, [-1, 3 * cnn_size, head_size, 1], [2, 2, 1, cnn_size],
                                       [1, 1, 1, 1],
                                       [-1, 3 * cnn_size, head_size * cnn_size], 'conE_cnn_d', 'SAME',
                                       train_summaries,
                                       is_trainsum_flag)


        with tf.name_scope('fully_connected'):
            self.user_cons = tf.concat(const_list + flen_list, 1)
            user_len = const_len + f_len

            if is_trainsum_flag:
                train_summaries.append(tf.summary.histogram('user_embedding', self.user_cons))

            target_embd = tf.concat([local_embd, conE_cnn_layer_t], 1)

            in_len = hidden_size * cnn_size + hidden_size

            # [B,E]
            layer1 = dense_layer(target_embd, in_len, in_len / 4, 'layer1', train_summaries, is_trainsum_flag,
                                 self.parameter, is_training=self.is_training)
            layer2 = dense_layer(layer1, in_len / 4, in_len / 8, 'layer2', train_summaries, is_trainsum_flag,
                                 self.parameter, is_training=self.is_training)
            layer3 = dense_layer(tf.concat([layer2, self.user_cons], 1), in_len / 8 + user_len, hidden_size, 'layer3',
                                 train_summaries, is_trainsum_flag, self.parameter, is_training=self.is_training)

            user_embeding_t = layer3

            # tag_embd [B,E]  self.output_layer: [B,1]
            self.output_layer = tf.reduce_sum(tf.multiply(user_embeding_t, tag_embd), axis=1, keep_dims=True)

            #tag_embd_gneg [B,T,E]
            layer_label_gneg = tag_embd_gneg

            #self.output_layer_gneg [B,T]
            self.output_layer_gneg = tf.reduce_sum(tf.multiply(tf.expand_dims(user_embeding_t, 1), layer_label_gneg), axis=-1)

            layer_label_recall = tag_embd_recall
            self.output_layer_recall = tf.reduce_sum(tf.multiply(tf.expand_dims(user_embeding_t, 1), layer_label_recall),
                                                   axis=-1)

            #[vocab_size,E]
            mp_tag = tag_embd_mp(self.feature_info_list)
            #[vocab_size,E]
            mp_tag_logits = tf.matmul(user_embeding_t, tf.transpose(mp_tag))
            #[vocab_size,E]
            self.sigmoid_mp_tag_logits = tf.nn.sigmoid(mp_tag_logits, name='sigmoid_mp_tag')

            mp_tag1w = tag_embd_mp(self.feature_info_list, range_cnt=10000)
            mp_tag_logits1w = tf.matmul(user_embeding_t, tf.transpose(mp_tag1w))
            self.sigmoid_mp_tag_logits1w = tf.nn.sigmoid(mp_tag_logits1w, name='sigmoid_mp_tag1w')

            mp_tag_id = tag_embd_mp(self.feature_info_list, tagids=self.tagids)
            mp_tag_logits_id = tf.matmul(user_embeding_t, tf.transpose(mp_tag_id))
            self.sigmoid_mp_tag_logits_id = tf.nn.sigmoid(mp_tag_logits_id, name='sigmoid_mp_tag_id')

        with tf.name_scope('train_op'):
            self.sigmoid_v = tf.nn.sigmoid(self.output_layer, name='sigmoid_v')
            self.sigmoid_gneg = tf.nn.sigmoid(self.output_layer_gneg, name='sigmoid_gneg')
            self.sigmoid_recall = tf.nn.sigmoid(self.output_layer_recall, name='sigmoid_gneg')

            self.cross_entropy = tf.reduce_sum(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=self.label_input, logits=self.output_layer))
            self.cross_entropy_gneg = tf.losses.sigmoid_cross_entropy(self.label_input_gneg, self.output_layer_gneg)

            self.train_op = optimizer_fuc(self.cross_entropy, self.global_step, learning_rate=self.lr)
            self.train_op_gneg = optimizer_fuc(self.cross_entropy_gneg, self.global_step, learning_rate=self.lr)

            train_summaries.append(tf.summary.scalar("loss_train", self.cross_entropy))

        with tf.name_scope('auc'):
            self.auc = auc_calculate(self.targets_input_gneg, self.sigmoid_gneg,
                                     input_name='train', summaries_list=train_summaries)
            self.auc_test = auc_calculate(self.targets_input, self.sigmoid_v,
                                          input_name='test', summaries_list=test_summaries)

        with tf.name_scope('recall'):
            self.recall = recall_calculate(self.targets_input_gneg, self.sigmoid_gneg,
                                          input_name='train', summaries_list=train_summaries)


            self.recall_test = recall_calculate(self.targets_input_recall, self.sigmoid_recall,
                                          input_name='test', summaries_list=None)

        with tf.name_scope('basic_info'):
            self.saver = tf.train.Saver()
            self.init_op = tf.global_variables_initializer()
            self.local_init_op = tf.local_variables_initializer()
            self.get_features_test = get_input_batches(test_file_list, self.batch_size)
            self.train_summary_op = tf.summary.merge(train_summaries)
            self.test_summary_op = tf.summary.merge(test_summaries)

    def get_feed_dict(self, feature_tensor, f_label, is_training=False):
        res_dict = {self.dropout_keep_prob: self.dp_val}

        for index, info in enumerate(self.feature_info_list):
            res_dict[info['feature_input']] = [x[index] for x in feature_tensor]

        a = np.array([[x[0]] for x in f_label])
        b = np.array([[x[1]] for x in f_label])
        c = np.array([[x[2]] for x in f_label])
        res_dict[self.label_input] = a
        res_dict[self.labelid] = b
        res_dict[self.labelLoid] = c
        res_dict[self.targets_input] = np.array(a, dtype=bool)

        res_dict[self.is_training] = is_training

        return res_dict

    def get_feed_dict_gneg(self, feature_tensor, f_label, is_training=False):
        res_dict = {self.dropout_keep_prob: self.dp_val}

        for index, info in enumerate(self.feature_info_list):
            res_dict[info['feature_input']] = [x[index] for x in feature_tensor]

        a = np.array([x[0] for x in f_label])
        b = np.array([x[1] for x in f_label])
        c = np.array([[x[2]] for x in f_label])
        res_dict[self.label_input_gneg] = a
        res_dict[self.labelid_gneg] = b
        res_dict[self.labelLoid] = c
        res_dict[self.targets_input_gneg] = np.array(a, dtype=bool)

        res_dict[self.is_training] = is_training

        return res_dict


    def get_feed_dict_recall(self, feature_tensor, f_label, is_training=False):
        res_dict = {self.dropout_keep_prob: self.dp_val}

        for index, info in enumerate(self.feature_info_list):
            res_dict[info['feature_input']] = [x[index] for x in feature_tensor]

        a = np.array([x[0] for x in f_label])
        b = np.array([x[1] for x in f_label])
        c = np.array([[x[2]] for x in f_label])
        res_dict[self.label_input_recall] = a
        res_dict[self.labelid_recall] = b
        res_dict[self.labelLoid] = c
        res_dict[self.targets_input_recall] = np.array(a, dtype=bool)

        res_dict[self.is_training] = is_training

        return res_dict