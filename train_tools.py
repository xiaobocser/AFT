import math

from libs.configs import cfgs
from data_process_tools import *
from libs.help_utils.util import *


def negative_sample(candidates):
    D = np.shape(candidates)[0]
    r = random.uniform(0, 1)
    index = int(math.ceil(math.pow(D + 1, r))) - 2
    return candidates[index], index

def sample_candidates(term_embedding_table, term_biases, begin_index):
    sampled_embeddings = []
    sampled_biases = []
    sampled_ids = []
    sampled_inverse_ids = {}
    for i in range(cfgs.generate_sample_num):
        embed, idx = negative_sample(term_embedding_table)
        while idx in sampled_ids:
            embed, idx = negative_sample(term_embedding_table)
        sampled_embeddings.append(embed)
        sampled_biases.append(term_biases[idx])

        sampled_inverse_ids[idx + begin_index] = len(sampled_ids)
        # sampled_ids.append(idx+begin_index)
        sampled_ids.append(idx)
        if i > len(term_biases) / 2: break

    sampled_ids = [x + begin_index for x in sampled_ids]

    _generate_sample_num = len(sampled_biases)

    sampled_embeddings = np.array(sampled_embeddings).reshape([_generate_sample_num, cfgs.hidden_size])
    sampled_biases = np.array(sampled_biases).reshape([_generate_sample_num, 1])
    sampled_ids = np.array(sampled_ids).reshape([_generate_sample_num, 1])

    return sampled_embeddings, sampled_biases, sampled_ids, sampled_inverse_ids

def labels_predict(sess, model, train_batch, ground_truth, prob, gen_labels, target_channel,
                   label_inverse_map, pos_labels=None, merge_ratio=0.0):
    f_feature_g = load_data_generator(train_batch, len(gen_labels), withuin=True, generator=len(gen_labels[0]))

    f_label_g = []
    label_idx = []

    if pos_labels is not None and len(pos_labels) != len(gen_labels):
        return -1, False

    for i, x in enumerate(gen_labels):
        true_set = list(ground_truth[i])
        if pos_labels is not None and random.random() < merge_ratio:
            for _ in x:
                xx = random.choice(pos_labels[i])
                f_label_g.append([0.0, xx, float(target_channel)])
                label_idx.append(label_inverse_map[int(xx)])
        else:
            for k, xx in enumerate(x):
                try:
                    index = true_set.index(label_inverse_map[int(xx)])
                except:
                    index = -1
                else:
                    print("%s in ground truth! prob:%s" % (xx, prob[i][k]))
                f_label_g.append([0.0, xx, float(target_channel)])
                label_idx.append(label_inverse_map[int(xx)])

    if len(f_feature_g) != len(f_label_g):
        print('length error!! ' * 20, file=sys.stderr)
        print(len(f_feature_g), file=sys.stderr)
        print(len(f_label_g), file=sys.stderr)
        return 0, 0, 0

    f_feature = f_feature_g
    f_label = f_label_g

    _sigmoid_v= sess.run([model.sigmoid_v], feed_dict=model.get_feed_dict(f_feature, f_label, is_training=True))

    f_label = [x[1] for x in f_label]
    f_label = np.array(f_label).reshape([-1, len(gen_labels[0])])
    ret_score = np.array(_sigmoid_v).reshape([-1, len(gen_labels[0])])
    label_idx = np.array(label_idx).reshape([-1, len(gen_labels[0])])

    return f_label, ret_score, label_idx


def loss_train(sess, model, train_auc_list, train_recall_list, loss_list, train_batch,
               true_labels, gen_labels, target_channel, g_neg_cnt, g_pos_cnt):
    _sum_loss = None

    f_feature_g = load_data_generator(train_batch, len(gen_labels), withuin=True, generator=len(gen_labels[0]),
                                      pointwise=False)
    f_label_g, f_id, f_label = [], [], []
    for index, x in enumerate(gen_labels):
        f_label_g.append([[1] * g_pos_cnt + [0] * g_neg_cnt, true_labels[index].tolist() + gen_labels[index].tolist(),
                          float(target_channel)])

    if len(f_feature_g) != len(f_label_g):
        print('length error!! ' * 20, file=sys.stderr)
        print(len(f_feature_g), file=sys.stderr)
        print(len(f_label_g), file=sys.stderr)
        return 0, 0, 0


    _, step, _cross_entropy_gneg, _sigmoid_gneg = sess.run(
        [model.train_op_gneg, model.global_step, model.cross_entropy_gneg, model.sigmoid_gneg],
        feed_dict=model.get_feed_dict_gneg(f_feature_g, f_label_g, is_training=True))

    sigmoid_gneg, _auc, _recall = sess.run([model.sigmoid_gneg, model.auc,
                                            model.recall],feed_dict=model.get_feed_dict_gneg(f_feature_g, f_label_g))

    train_auc_list[0].append(_auc)
    train_recall_list[0].append(_recall)
    loss_list[0].append(_cross_entropy_gneg / len(f_label_g))

    ret_score = []
    for x in _sigmoid_gneg:
        ret_score.append(x[g_pos_cnt:])

    return step, _sum_loss, ret_score

def predict_test(sess, model, writer, step, test_auc_list, test_recall_list, index_dict, recall_topn=20):
    """Print Test AUC and loss for one batch in TestData"""
    lines = sess.run(model.get_features_test)
    f_feature, f_label, click_item = load_data(lines, model.batch_size, max_term_num=cfgs.max_term_num, withuin=True)

    test_auc, sigmoid_mp_tag_logits = sess.run([model.auc_test, model.sigmoid_mp_tag_logits],feed_dict=model.get_feed_dict(f_feature, f_label))

    test_auc_list[0].append(test_auc)
    avg_test_auc = sum(test_auc_list[0]) / len(test_auc_list[0])
    test_auc_list[1] = max(test_auc_list[1], avg_test_auc)

    cal_auc_out_1posNneg(avg_test_auc, None, None, step, test_auc_list[1], 'test AUC')

    recall_score = 0.0
    i = 0
    for index in range(0,len(sigmoid_mp_tag_logits),2):
        res_fil = [[x, sigmoid_mp_tag_logits[index][x]] for x in index_dict]
        topn_res = [i for i, val in sorted(res_fil, key=lambda v: v[1], reverse=True)[:recall_topn]]

        t = set(click_item[(int)(index / 2)]) & set(index_dict)
        recall_num = len(t & set(topn_res))

        recall_score += 0.0 if len(t) == 0 else float(recall_num) / min(len(t),recall_topn)
        i += 0 if len(t) == 0 else 1

    test_recall = 0.0 if i == 0 else recall_score / i

    test_recall_list[0].append(test_recall)
    avg_test_recall = sum(test_recall_list[0]) / len(test_recall_list[0])
    test_recall_list[1] = max(test_recall_list[1], avg_test_recall)

    cal_recall_out_1posNneg(avg_test_recall, None, None, step, test_recall_list[1], 'test recall')

def loss_output(loss_list, writer, step, train_auc_list, train_recall_list,_sum_loss):
    """Print training AUC and loss"""
    avg_loss = sum(loss_list[0])/len(loss_list[0])
    loss_list[1] = min(loss_list[1], avg_loss)

    avg_auc = sum(train_auc_list[0])/len(train_auc_list[0])
    train_auc_list[1] = max(train_auc_list[1], avg_auc)

    avg_recall = sum(train_recall_list[0]) / len(train_recall_list[0])
    train_recall_list[1] = max(train_recall_list[1], avg_recall)

    time_str = datetime.datetime.now().isoformat()
    print('=' * 10 + time_str + '=' * 10, file=sys.stderr)
    print('%d step, avg loss is %.5f, min loss is %.5f' % (step, avg_loss, loss_list[1]), file=sys.stderr)
    print('%d step, avg train auc is %.5f. max train auc is %.5f' % (step, avg_auc, train_auc_list[1]), file=sys.stderr)
    print('%d step, avg train recall is %.5f. max train recall is %.5f' % (step, avg_recall, train_recall_list[1]), file=sys.stderr)

    if writer is not None and _sum_loss is not None:
        writer.add_summary(_sum_loss, step)


def out_param(sess, model, predict_data_save_path):
    param = sess.run([model.parameter])[0]
    f_w = open(predict_data_save_path + '_param', 'a+')
    for key, val in param.iteritems():
        print(key, file=sys.stderr)
        if key in ['l1cate_matrix', 'l2cate_matrix', 'tag_matrix', 'topic_matrix']:
            for index, vv in enumerate(val):
                f_w.write('%s_%d\t%s\n' % (key, index, ' '.join([str(x) for x in vv.tolist()])))
        else:
            f_w.write('%s\t%s\n' % (key, ' '.join([str(x) for x in flatten(val.tolist())])))

    f_w.close()