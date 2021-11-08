import math
from libs.help_utils.metrics import auc_fun
import numpy as np
import six
import tensorflow as tf
from libs.help_utils import util

from libs.networks.model_tools import *

class BehaviorModel(object):
    def __init__(self,
                 vocab_size,
                 # validate_files,
                 hidden_size=64,
                 ffn_size=128,
                 num_attention_head=6,
                 num_hidden_layers=4,
                 attention_drop_prob=0.1,
                 hidden_drop_prob=0.1,
                 max_term_num=50,
                 max_channel_embed_size=7,
                 initializer_range=0.1,
                 hidden_act="gelu",
                 learning_rate=0.001,
                 pos_num=5,
                 neg_num=25,
                 reward_num=5,
                 batch_size=256):

        train_summaries = []

        # Model params
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.profile_hidden_size = 8
        self.ffn_size = ffn_size
        self.num_attention_head = num_attention_head
        self.num_hidden_layers = num_hidden_layers
        self.attention_drop_prob = attention_drop_prob
        self.hidden_drop_prob = hidden_drop_prob
        self.max_channel_embed_size = max_channel_embed_size
        self.initializer_range = initializer_range
        self.hidden_act = hidden_act
        self.max_term_num = max_term_num
        self.batch_size = batch_size

        self.learning_rate = learning_rate
        self.global_step = tf.Variable(0, trainable=False, name="global_step")
        self.reward_num = reward_num
        self.g_params = []
        self.grad_clip = 5.0
        # tf.Variable(0, trainable=False, name="global_step")
        self.test_step = tf.Variable(0, trainable=False, name="test_step")
        self.validate_step = tf.Variable(0, trainable=False, name="validate_step")
        self.add_test_step = tf.assign_add(self.test_step, 1)
        self.add_validate_step = tf.assign_add(self.validate_step, 1)

        # Input features
        self.uin = tf.placeholder(tf.string, shape=[None, 1], name="uin")
        self.term_ids = tf.placeholder(tf.int64, shape=[None, max_channel_embed_size, max_term_num], name="term_ids")
        self.term_mask = tf.placeholder(tf.int64, shape=[None, max_channel_embed_size, max_term_num], name="term_mask")
        self.masked_channel = tf.placeholder(tf.int64, shape=[None, 1], name="masked_channel")
        self.profile_gender = tf.placeholder(tf.int64, shape=[None, 1], name="profile_gender")
        self.profile_age = tf.placeholder(tf.int64, shape=[None, 1], name="profile_age")
        self.profile_region = tf.placeholder(tf.int64, shape=[None, 1], name="profile_region")
        self.profile_elite = tf.placeholder(tf.int64, shape=[None, 1], name="profile_elite")
        self.label = tf.placeholder(tf.int64, shape=[None, pos_num + neg_num], name="label")
        self.g_labels = tf.placeholder(tf.int64, shape=[None, reward_num], name="g_labels")
        self.g_labels_idx = tf.placeholder(tf.int64, shape=[None, reward_num], name="g_labels_idx")
        self.rewards = tf.placeholder(tf.float32, shape=[None, reward_num], name="rewards")
        self.term_init_map = tf.placeholder(
                dtype=tf.float32, shape=[self.vocab_size, self.hidden_size], name="term_init_map")

        self.sampled_embeds = tf.placeholder(tf.float32, shape=[None, hidden_size], name="sampled_embeds")
        self.sampled_biases = tf.placeholder(tf.float32, shape=[None, 1], name="sampled_biases")
        self.sampled_idx = tf.placeholder(tf.float32, shape=[None, 1], name="sampled_idx")
        self.ground_truth = tf.placeholder(tf.int64, shape=[None, max_term_num + pos_num], name="ground_truth")
        self.alpha = tf.placeholder(tf.float32, name="alpha")

        self.channel_embeddings = tf.get_variable(
            name="channel_embedding_table",
            shape=[self.max_channel_embed_size, self.hidden_size],
            initializer=create_initializer(self.initializer_range))

        self.pos_num = pos_num
        self.neg_num = neg_num
        self.is_training = tf.placeholder(tf.bool, name="is_training")
        self.debug_tensors = {}
        self.debug_keys = []

        with tf.variable_scope("embeddings"):
            self.term_embedding_table = tf.get_variable(
                name="term_embedding_table",
                shape=[vocab_size, self.hidden_size],
                initializer=create_initializer(self.initializer_range))
            self.item_embed_init = self.term_embedding_table.assign(self.term_init_map)

            channel_pooling = []
            self.masks = []
            for i in range(self.max_channel_embed_size):
                # `channel_input`: [B, T]
                channel_input = tf.squeeze(tf.slice(
                    self.term_ids, [0, i, 0], [-1, 1, max_term_num]), axis=1)

                #domain-specific masked
                channel_input *= (1 - tf.to_int64(tf.equal(self.masked_channel, i)) * tf.to_int64(tf.random.uniform(tf.shape(channel_input), 0, 1) < 0.15))

                #channel_mask [B, T]
                channel_mask = tf.to_float(tf.abs(channel_input) > 0)

                self.masks.append(tf.reduce_max(channel_mask, axis = 1, keep_dims=True))

                # `channel_embedding_output`: [B, T, E]
                channel_embedding_output = tf.nn.embedding_lookup(
                    self.term_embedding_table,
                    channel_input,
                    name="channel_embedding_output_%s" % i)

                # Channel mask
                attention_mask = create_attention_mask_from_input_mask(
                    channel_embedding_output, channel_mask)

                # `channel_layer`: [B, T, N * H]
                channel_layer = attention_layer(
                    from_tensor=channel_embedding_output,
                    to_tensor=channel_embedding_output,
                    is_training=self.is_training,
                    attention_mask=attention_mask,
                    num_attention_heads=1,
                    size_per_head=hidden_size,
                    attention_probs_dropout_prob=self.attention_drop_prob,
                    initializer_range=initializer_range,
                    batch_size=batch_size,
                    from_seq_length=max_term_num,
                    to_seq_length=max_term_num,
                    suffix="channel_attention_%s" % i)

                channel_attention_output = tf.layers.dense(
                    channel_layer,
                    hidden_size,
                    name="channel_attention_output_%s" % i,
                    kernel_initializer=create_initializer(initializer_range))
                channel_attention_pool = tf.reduce_sum(channel_attention_output, axis=1)
                channel_attention_pool = layer_norm_and_dropout(
                    channel_attention_pool, self.hidden_drop_prob, is_training=self.is_training)

                # [B,E]
                channel_pooling.append(channel_attention_pool)

            # `term_pooling`: [B, D, E]
            self.term_pooling = tf.reshape(
                tf.concat(channel_pooling, axis=1), [-1, self.max_channel_embed_size, hidden_size])

            self.term_pooling += self.channel_embeddings
            self.term_pooling = tf.multiply(
                self.term_pooling, tf.expand_dims(tf.concat(self.masks, axis=1), -1))

        with tf.variable_scope("cross"):
            # `attention_out`: [B, D, E]
            merge_attention_layer = attention_layer(
                from_tensor=self.term_pooling,
                to_tensor=self.term_pooling,
                is_training=self.is_training,
                attention_mask=None,
                num_attention_heads=num_attention_head,
                size_per_head=hidden_size,
                attention_probs_dropout_prob=self.attention_drop_prob,
                initializer_range=initializer_range,
                batch_size=batch_size,
                from_seq_length=max_channel_embed_size,
                to_seq_length=max_channel_embed_size,
                suffix="encoder_attention")

            # `merge_attention_output`: [B, D, E]
            merge_attention_output = tf.layers.dense(
                merge_attention_layer,
                hidden_size,
                kernel_initializer=create_initializer(initializer_range))
            merge_attention_output = dropout(merge_attention_output, hidden_drop_prob, is_training=self.is_training)
            self.attention_out = layer_norm(merge_attention_output + self.term_pooling)


        with tf.variable_scope("Aggregator"):

            mask_one_hot = tf.one_hot(tf.squeeze(self.masked_channel, axis=-1),
                                      depth=max_channel_embed_size, dtype=tf.float32)
            # Mask target_terms: [B, 1, E]
            target_terms = tf.reduce_sum(
                tf.multiply(self.term_pooling, tf.reshape(mask_one_hot, [-1, max_channel_embed_size, 1])), 1, keep_dims=True)

            # Mask target_channel_terms: [B,1,E]
            target_channel_terms = tf.reduce_sum(
                tf.multiply(tf.reshape(self.channel_embeddings,[1,max_channel_embed_size,hidden_size]), tf.reshape(mask_one_hot, [-1, max_channel_embed_size, 1])), 1,
                keep_dims=True)
            target_terms += target_channel_terms

            # `decode_attention_layer`: [B, 1, E]
            decode_attention_layer = attention_layer(
                from_tensor=target_terms,
                to_tensor=self.attention_out,
                is_training=self.is_training,
                attention_mask=None,
                num_attention_heads=1,
                size_per_head=hidden_size,
                attention_probs_dropout_prob=self.attention_drop_prob,
                initializer_range=initializer_range,
                batch_size=batch_size,
                from_seq_length=1,
                to_seq_length=max_channel_embed_size,
                suffix="decoder_attention")

        concat_layer = tf.concat([decode_attention_layer, target_terms], axis=2)
        self.concat_layer = layer_norm(concat_layer)

        self.ffn_1 = tf.layers.dense(
            tf.reshape(self.concat_layer, [-1, hidden_size + hidden_size]),
            ffn_size,
            activation=gelu,
            name="ffn_layer_1",
            kernel_initializer=create_initializer(self.initializer_range))

        self.mask_embedding = tf.layers.dense(
            self.ffn_1,
            hidden_size,
            activation=gelu,
            name="ffn_layer_2",
            kernel_initializer=create_initializer(self.initializer_range))
        self.mask_embedding = tf.expand_dims(self.mask_embedding, 1)

        self.mask_embedding = decode_attention_layer

        self.label_bias_table = tf.get_variable(
            "output_bias_table",
            shape=[self.vocab_size],
            initializer=tf.zeros_initializer())

        # Label bias: [B, (P + N)]
        label_bias = tf.gather(self.label_bias_table, self.label)
        # Label weight: [B, (P + N)]
        label_weights = tf.concat(
            [tf.constant(1.0, shape=[batch_size, self.pos_num]),
             tf.constant(0.0, shape=[batch_size, self.neg_num])], axis=1)

        # label embedding: [B, (P + N), E]
        label_embedding = tf.nn.embedding_lookup(
            self.term_embedding_table,
            self.label,
            name="label_embedding")
        self.logits, self.logits_sigmoid, self.loss, self.auc = get_masked_lm_output(
            self.mask_embedding, label_embedding, label_bias, label_weights)
        train_summaries.append(tf.summary.scalar("loss", self.loss))
        train_summaries.append(tf.summary.scalar("auc", self.auc))

        with tf.variable_scope("pretrain"):
            self.train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate)\
                .minimize(self.loss, global_step=self.global_step, name="train_op")

        with tf.variable_scope("train_adversarial"):
            self.truth_one_hot = tf.reduce_min(
                tf.one_hot(self.ground_truth, tf.shape(self.sampled_embeds)[0], 0.0, 1.0), axis=1)
            self.generate_labels, self.generate_labels_score, self.prob, self.gen_idx = self.generate_label(
                self.mask_embedding, self.sampled_embeds, self.sampled_biases, self.sampled_idx,
                self.ground_truth, self.reward_num)

            # [B,reward_num]
            self.g_predictions = tf.reduce_sum(
                tf.multiply(tf.one_hot(self.g_labels_idx, tf.shape(self.sampled_embeds)[0], 1.0, 0.0),
                            tf.expand_dims(self.prob, 1)), 2)

            # [B,reward_num,E]
            self.g_embeds = tf.tensordot(
                tf.one_hot(self.g_labels_idx, tf.shape(self.sampled_embeds)[0], 1.0, 0.0),
                self.sampled_embeds, axes=[2, 0])

            self.mmd = util.mmd_loss(
                tf.reshape(self.g_embeds, [-1, hidden_size]),
                tf.reshape(tf.slice(label_embedding, [0, 0, 0], [-1, pos_num, -1]), [-1, hidden_size]),
                self.alpha)
            self.rewards_balance = (self.rewards) / (1.0 * self.reward_num)
            self.g_loss = -tf.reduce_mean(
                    tf.log(self.g_predictions)
                    * tf.reshape(self.rewards_balance - self.mmd, [-1, self.reward_num]), axis=1)

            # self.g_loss = -(tf.reduce_mean(
            #     self.g_predictions
            #     * tf.reshape(self.rewards_balance, [-1, self.reward_num]), axis=1) + self.mmd)

            g_opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            grads_pair = g_opt.compute_gradients(self.g_loss)
            grads_pair = [x for x in grads_pair if x[0] is not None]
            grads, _ = tf.clip_by_global_norm([x[0] for x in grads_pair], self.grad_clip)
            grads_pair = [x for x in zip(grads, [g[1] for g in grads_pair])]
            self.g_updates = g_opt.apply_gradients(grads_pair, global_step=self.global_step, name="g_updates")

        self.saver = tf.train.Saver(max_to_keep=1)
        self.init_op = tf.global_variables_initializer()
        self.local_init_op = tf.local_variables_initializer()
        self.train_summary = tf.summary.merge(train_summaries)

    def generate_label(self, gen_output, term_table, term_biases, sampled_idx, ground_truth, reward_num=5):
        """
        Args:
            gen_output: (B,1,E)
            term_table: (T,E)   #T is sample_num
            term_biases: (T,1)
            sampled_idx: (T,1)
            ground_truth: (B,N+P) #N is max_term_num
            reward_num: select top reward_num

        Returns:

        """
        # `truth_onehot`: [B, T]
        truth_one_hot = tf.reduce_min(tf.one_hot(ground_truth, tf.shape(term_table)[0], 0.0, 1.0), axis=1)

        # `gen_output`: [B,E]
        gen_output = tf.squeeze(gen_output, axis=1)
        # `dot`: [B, T]
        dot = tf.matmul(gen_output, term_table, transpose_b=True) + tf.reshape(term_biases, [-1])
        prob = tf.nn.softmax(dot)
        prob = tf.multiply(prob, truth_one_hot)  # click item set 0
        #log_prob = tf.log(prob)

        # [B, reward_num]
        gen_idx = tf.cast(tf.multinomial(tf.log(prob), reward_num), tf.int64)  # [B,reward_num]  number in sample
        gen_labels = tf.cast(tf.reduce_sum(
            tf.multiply(tf.one_hot(gen_idx, tf.shape(term_table)[0], 1.0, 0.0),  # [B,reward_num]  number in whole sample
                        tf.reshape(sampled_idx, [-1])), 2), tf.int64)
        pred = tf.reduce_sum(
            tf.multiply(tf.one_hot(gen_idx, tf.shape(term_table)[0], 1.0, 0.0),  # [B,reward_num]  probability
                        tf.expand_dims(prob, 1)), 2)

        # [B,sample_num], [B,sample_num], [B,T], [B,sample_num]
        return gen_labels, pred, prob, gen_idx

















