# -*- coding:utf-8 -*-

from __future__ import print_function
# reload(sys)
# sys.setdefaultencoding('utf-8')

# from compiler.ast import flatten
import tensorflow.contrib.slim as slim

from libs.networks.model_graph import ModelGraph
from libs.networks.model_generator import BehaviorModel
from train_tools import *
from tqdm import tqdm

tf.app.flags.DEFINE_string("root_path", "", "Train root path")
tf.app.flags.DEFINE_string("suffix", "", "Model suffix")
tf.app.flags.DEFINE_string("init_g_checkpoint", "", "Generator model checkpoint")
tf.app.flags.DEFINE_string("init_d_checkpoint", "", "Discriminator model checkpoint")
tf.app.flags.DEFINE_string("save_path", "", "Save path")
tf.app.flags.DEFINE_integer("max_batch", 2000, "Max training batch")
tf.app.flags.DEFINE_integer("is_debug", 0, "Is debugging")
tf.app.flags.DEFINE_integer("load_pretrain", 0, "If load latest model")
tf.app.flags.DEFINE_integer("show_n_batches", 0, "Show every n batches")
tf.app.flags.DEFINE_integer("test_n_batches", 0, "Test every n batches")
tf.app.flags.DEFINE_integer("save_n_batches", 5000, "Save every n batches")

tf.app.flags.DEFINE_string("need_finetuning", '', "model finetuning arg.")
FLAGS = tf.app.flags.FLAGS

discriminator_save_path = cfgs.OUTPUT_PATH + "/trained_model/discriminator" if not FLAGS.save_path else FLAGS.save_path + "/trained_model/discriminator"

whole_model_save_path = cfgs.OUTPUT_PATH + "/trained_model/whole_model" if not FLAGS.save_path else FLAGS.save_path + "/trained_model/whole_model"

generator_pretrain_path = cfgs.OUTPUT_PATH + "/trained_model/pretrain_generator" if not FLAGS.save_path else FLAGS.save_path + "/trained_model/pretrain_generator"

discriminator_pretrain_path = cfgs.OUTPUT_PATH + "/trained_model/pretrain_discriminator" if not FLAGS.save_path else FLAGS.save_path + "/trained_model/pretrain_discriminator"


def train(sess, sv, discriminator, generator, saver, next_batch):
    #global loss_step, save_step

    next_train_batch_list, next_test_batch_list = next_batch

    print('begin session', file=sys.stdout)

    d_step, g_step, save_flag = 0, 0, False
    train_auc_list, train_recall_list,test_auc_list, test_recall_list, loss_list, _sum_loss = [[], 0.0], [[], 0.0], [[], 0.0], [[], 0.0], [[], 1.0e10], None

    writer = tf.summary.FileWriter(cfgs.sum_dir, sess.graph)

    try:
        #pretrain
        for _ in tqdm(list(range(cfgs.pretrain_iter_g))):
            channnnl = [3, 19]
            for target_channel in channnnl:
                next_train_batch = next_train_batch_list[target_channel]
                train_batch = sess.run(next_train_batch)
                feature_map = load_batch_data(train_batch, cfgs.batch_size, pos_num=cfgs.pos_num,
                                              neg_num=cfgs.neg_num,
                                              max_channel_embed_size=cfgs.max_channel_embed_size,
                                              max_term_num=cfgs.max_term_num)

                feed_dict = {
                    generator.uin: feature_map["uin"],
                    generator.term_ids: feature_map["term_ids"],
                    generator.term_mask: feature_map["term_mask"],
                    generator.masked_channel: np.ones([cfgs.batch_size, 1]) * target_channel,
                    generator.label: feature_map["labels"],
                    generator.is_training: True
                }

                sess.run([generator.train_op], feed_dict)
        save_model_out(sess, generator, generator_pretrain_path + '/generator_pretrain.%s' % cfgs.pretrain_iter_g, save_flag)
        #generator.saver.restore(sess, generator_pretrain_path + '/generator_pretrain.%s' % cfgs.pretrain_iter_g)
        sess.run(generator.global_step.initializer)

        for _ in range(cfgs.pretrain_iter_d):
            list_check(loss_list, train_auc_list, train_recall_list, test_auc_list, test_recall_list)

            # Train generator
            channnnl = [3, 19]
            for target_channel in channnnl:
                next_train_batch = next_train_batch_list[target_channel]
                for i in range(cfgs.D_steps):
                    train_batch = sess.run(next_train_batch)
                    feature_map = load_batch_data(train_batch, cfgs.batch_size, pos_num=cfgs.pos_num,
                                                  neg_num=cfgs.neg_num,
                                                  max_channel_embed_size=cfgs.max_channel_embed_size,
                                                  max_term_num=cfgs.max_term_num)
                    d_step, _sum_loss, ret_score = loss_train(
                            sess, discriminator, train_auc_list, train_recall_list, loss_list, train_batch,
                            feature_map["labels"][:, :cfgs.pos_num], feature_map["labels"][:, cfgs.pos_num:],
                            target_channel, cfgs.neg_num, cfgs.pos_num)

            if _sum_loss is False: continue

            if d_step % cfgs.loss_step == 0:
                print("d_loss:", file=sys.stdout)
                loss_output(loss_list, writer, d_step, train_auc_list, train_recall_list, _sum_loss)

            if d_step % cfgs.test_step == 0 and d_step > 0:
                index_dict = [int(line.rstrip('\n').split('\t')[1]) for line in open(cfgs.label_file)]
                predict_test(sess, discriminator, writer, d_step, test_auc_list, test_recall_list, index_dict)
        save_model_out(sess, discriminator, discriminator_pretrain_path + '/discriminator_pretrain.%s' % cfgs.pretrain_iter_d, save_flag)
        #discriminator.saver.restore(sess, discriminator_save_path + '/discriminator_pretrain.%s' % cfgs.pretrain_iter_d)
        sess.run(discriminator.global_step.initializer)

        while not sv.should_stop() and g_step < FLAGS.max_batch:
            list_check(loss_list, train_auc_list, train_recall_list, test_auc_list, test_recall_list)

            # Train generator
            channnnl = [3, 19]
            for target_channel in channnnl:
                next_train_batch = next_train_batch_list[target_channel]

                print("Resample", file=sys.stdout)
                term_embedding_table = sess.run(generator.term_embedding_table)[
                                       cfgs.tag_list[target_channel][0]: cfgs.tag_list[target_channel][1], :]
                term_biases = sess.run(generator.label_bias_table)[
                              cfgs.tag_list[target_channel][0]: cfgs.tag_list[target_channel][1]]
                sampled_embeddings, sampled_biases, sampled_ids, label_inverse_map = sample_candidates(
                    term_embedding_table, term_biases, cfgs.tag_list[target_channel][0])

                print("G steps %d, for ch:%d, sample size: %d" % (g_step, target_channel, len(sampled_biases)),
                      file=sys.stdout)

                for i in range(cfgs.G_steps):
                    # timea = time.time()
                    train_batch = sess.run(next_train_batch)
                    feature_map = load_batch_data(train_batch, cfgs.batch_size, pos_num=cfgs.pos_num,
                                                  neg_num=cfgs.neg_num,
                                                  max_channel_embed_size = cfgs.max_channel_embed_size,
                                                  max_term_num = cfgs.max_term_num)
                    ground_truth = np.zeros([cfgs.batch_size, cfgs.max_term_num + cfgs.pos_num])
                    for m in range(cfgs.batch_size):
                        ground_truth[m][:cfgs.max_term_num] = np.array(list(map(
                            lambda x: label_inverse_map[int(x)] if int(x) in label_inverse_map else 0,
                            feature_map["term_ids"][m][target_channel])))
                        ground_truth[m][cfgs.max_term_num:cfgs.max_term_num + cfgs.pos_num] = np.array(list(map(
                            lambda x: label_inverse_map[int(x)] if int(x) in label_inverse_map else 0,
                            feature_map["labels"][m, :cfgs.pos_num])))

                    feed_dict = {
                        generator.uin: feature_map["uin"],
                        generator.term_ids: feature_map["term_ids"],
                        generator.term_mask: feature_map["term_mask"],
                        generator.masked_channel: np.ones([cfgs.batch_size, 1]) * target_channel,
                        generator.ground_truth: ground_truth,
                        generator.sampled_embeds: sampled_embeddings,
                        generator.sampled_biases: sampled_biases,
                        generator.sampled_idx: sampled_ids,
                        generator.alpha: 0.0,
                        generator.label: feature_map["labels"],
                        generator.is_training: True
                    }

                    gen_labels, pred, prob, truth_one_hot, gen_idx, mask_embedding, g_step = sess.run(
                        [generator.generate_labels, generator.generate_labels_score, generator.prob,
                         generator.truth_one_hot, generator.gen_idx,
                         generator.mask_embedding, generator.global_step], feed_dict)

                    # Get rewards
                    try:
                        labels, ret_score, labels_idx = labels_predict(
                                sess, discriminator, train_batch, ground_truth, prob, gen_labels, target_channel,
                                label_inverse_map, pos_labels=list(feature_map["labels"][:, :cfgs.d_pos_num]), merge_ratio=0.0)
                    except Exception as err:
                        print('!!! error !!!' * 20, file=sys.stderr)
                        print(err, file=sys.stderr)
                        continue

                    feed_dict[generator.g_labels_idx] = labels_idx
                    feed_dict[generator.rewards] = ret_score
                    #feed_dict[generator.label] = feature_map["labels"]

                    _, prob, rewards, mmd_loss, g_loss, g_predictions, rewards_balance, g_step = sess.run(
                            [generator.g_updates, generator.prob, generator.rewards_balance, generator.mmd,
                             generator.g_loss, generator.g_predictions,generator.rewards_balance,
                             generator.global_step],
                            feed_dict)

                    feed_dict[generator.masked_channel] = feature_map["masked_channel"]
                    _train_sum = sess.run(generator.train_summary, feed_dict)
                    writer.add_summary(_train_sum, g_step)
                    _auc, _loss = sess.run([generator.auc, generator.loss], feed_dict)

                    if i % 10 == 0:
                        print("G step: %s, D step: %d, g_gan_loss:%s, MMD loss:%s, g_loss:%s, auc:%s, g_predictions: %s, d_predictions: %s" %
                                  (g_step, d_step, np.mean(g_loss), mmd_loss, _loss, _auc, np.sum(g_predictions), np.sum(rewards_balance)), file=sys.stdout)

                # Train discriminator
                print("D steps begin for ch%d" % (target_channel), file=sys.stdout)
                for i in range(cfgs.D_steps):
                    train_batch = sess.run(next_train_batch)
                    feature_map = load_batch_data(train_batch, cfgs.batch_size, pos_num=cfgs.pos_num,
                                                  neg_num=cfgs.neg_num,
                                                  max_channel_embed_size=cfgs.max_channel_embed_size,
                                                  max_term_num=cfgs.max_term_num)

                    ground_truth = np.zeros([cfgs.batch_size, cfgs.max_term_num + cfgs.pos_num])
                    for m in range(cfgs.batch_size):
                        ground_truth[m][:cfgs.max_term_num] = np.array(list(map(
                            lambda x: label_inverse_map[int(x)] if int(x) in label_inverse_map else 0,
                            feature_map["term_ids"][m][target_channel])))
                        ground_truth[m][cfgs.max_term_num:cfgs.max_term_num + cfgs.pos_num] = np.array((list(map(
                            lambda x: label_inverse_map[int(x)] if int(x) in label_inverse_map else 0,
                            feature_map["labels"][m, :cfgs.pos_num]))))
                    feed_dict = {
                        generator.uin: feature_map["uin"],
                        generator.term_ids: feature_map["term_ids"],
                        generator.term_mask: feature_map["term_mask"],
                        generator.masked_channel: np.ones([cfgs.batch_size, 1]) * target_channel,
                        generator.sampled_embeds: sampled_embeddings,
                        generator.sampled_biases: sampled_biases,
                        generator.sampled_idx: sampled_ids,
                        generator.ground_truth: ground_truth,
                        generator.is_training: False
                    }
                    gen_labels, pred, g_step = sess.run(
                        [generator.generate_labels, generator.generate_labels_score, generator.global_step], feed_dict)

                    d_step, _sum_loss, ret_score = loss_train(
                            sess, discriminator, train_auc_list, train_recall_list, loss_list, train_batch,
                            feature_map["labels"][:, :cfgs.d_pos_num], gen_labels, target_channel, cfgs.reward_num,
                            cfgs.d_pos_num)


            if _sum_loss is False: continue

            if d_step % cfgs.loss_step == 0:
                print("d_loss:", file=sys.stdout)
                loss_output(loss_list, writer, d_step, train_auc_list, train_recall_list, _sum_loss)

            if d_step - cfgs.save_step >= FLAGS.save_n_batches:
                saver.save(sess, whole_model_save_path + '/model.%s' % d_step)
                save_model_out(sess, discriminator, discriminator_save_path + '/discriminator.%s' % d_step, save_flag)
                cfgs.save_step = d_step

            if d_step % cfgs.test_step == 0 and d_step > 0:
                index_dict = [int(line.rstrip('\n').split('\t')[1]) for line in open(cfgs.label_file)]
                predict_test(sess, discriminator, writer, d_step, test_auc_list, test_recall_list, index_dict)

    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')

    # loop end
    save_model_out(sess, discriminator, discriminator_save_path + '/discriminator', save_flag)

    sv.stop()


def main():
    train_file_list = [get_file_list(cfgs.train_data_file_list[i]) for i in cfgs.channels]
    test_file_list = [get_file_list(cfgs.test_data_file_list[i]) for i in cfgs.channels]

    tf.reset_default_graph()

    model = ModelGraph(
                       test_file_list[cfgs.test_domain],
                       test_pos_num=cfgs.test_pos_num,
                       test_neg_num=cfgs.test_neg_num,
                       is_trainsum_flag=True,
                       ffn_size=cfgs.ffn_size,
                       vocab_size=cfgs.vocab_size,
                       g_neg_cnt=cfgs.reward_num + cfgs.d_pos_num,
                       max_term_num=cfgs.max_term_num,
                       num_attention_head=cfgs.num_attention_head_d,
                       num_hidden_layers=cfgs.num_hidden_layers_d,
                       hidden_size = cfgs.hidden_size,
                       attention_drop_prob = cfgs.attention_drop_prob_d,
                       hidden_drop_prob=cfgs.hidden_drop_prob_d,
                       initializer_range = cfgs.initializer_range_d,
                       domain_cnt = len(cfgs.channels),
                       cnn_size = cfgs.cnn_size
                       )

    generator = BehaviorModel(
                vocab_size=cfgs.vocab_size,
                hidden_size=cfgs.hidden_size,
                ffn_size=cfgs.ffn_size,
                num_attention_head=cfgs.num_attention_head_g,
                num_hidden_layers=cfgs.num_hidden_layers_g,
                attention_drop_prob=cfgs.attention_drop_prob_g,
                hidden_drop_prob=cfgs.hidden_drop_prob_g,
                max_term_num=cfgs.max_term_num,
                max_channel_embed_size=cfgs.max_channel_embed_size,
                initializer_range=cfgs.initializer_range_g,
                hidden_act="gelu",
                learning_rate=cfgs.learning_rate,
                pos_num=cfgs.pos_num,
                neg_num=cfgs.neg_num,
                batch_size=cfgs.batch_size,
                reward_num=cfgs.reward_num)

    next_train_batch_list = [get_input_batches(train_file_list[i], cfgs.batch_size) for i in cfgs.channels]
    next_test_batch_list = [get_input_batches(test_file_list[i], cfgs.batch_size) for i in cfgs.channels]
    next_batch = (next_train_batch_list, next_test_batch_list)

    print("G Model inited", file=sys.stdout)
    tvars = tf.trainable_variables()
    init_g_checkpoint = FLAGS.init_g_checkpoint

    print("**** Dan Trainable Variables ****")
    inclusion = []
    if init_g_checkpoint != "":
        print("  G model restore  " * 20, file=sys.stdout)
        (assignment_map, initialized_variable_names
         ) = get_assignment_map_from_checkpoint(tvars, init_g_checkpoint)
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
                inclusion.append(var.name)
            print("  name = %s, shape = %s%s" % (var.name, var.shape, init_string))
        inception_except_logits = slim.get_variables_to_restore(include=inclusion)
        init_fn_g = slim.assign_from_checkpoint_fn(FLAGS.init_g_checkpoint, inception_except_logits)
    else:
        init_fn_g = None

    g_init_op = tf.global_variables_initializer()
    saver = tf.train.Saver(max_to_keep=1)

    if FLAGS.need_finetuning == "need":
        print("---D model restore---" * 20, file=sys.stdout)
        variables_to_restore = tf.contrib.slim.get_variables_to_restore()
        model_file = tf.train.latest_checkpoint(cfgs.save_model_dir)
        init_restore = tf.contrib.slim.assign_from_checkpoint_fn(model_file,
                                                                 variables_to_restore,
                                                                 ignore_missing_vars=True)
        sv = tf.train.Supervisor(logdir="%s/train_logs" % cfgs.log_dir,
                                 init_op=g_init_op,
                                 summary_op=None,
                                 saver=saver,
                                 save_model_secs=0,
                                 global_step=model.global_step,
                                 init_fn=init_restore)

    else:
        sv = tf.train.Supervisor(logdir="%s/train_logs" % cfgs.log_dir,
                                 init_op=g_init_op,
                                 summary_op=None,
                                 saver=model.saver,
                                 save_model_secs=0,
                                 global_step=model.global_step)

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True

    with sv.managed_session(config=config) as sess:
        sess.run(model.local_init_op)
        sess.run(generator.init_op)
        if init_fn_g:
            init_fn_g(sess)
        if FLAGS.need_finetuning == "need":
            init_restore(sess)
        sess.run(model.global_step.initializer)
        train(sess, sv, model, generator, saver, next_batch)

if __name__ == '__main__':
    main()



