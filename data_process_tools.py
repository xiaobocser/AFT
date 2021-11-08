# encoding=utf-8

from __future__ import print_function
import tensorflow as tf
from tensorflow.contrib import layers
import numpy as np
import sys
import os
import random
import datetime
import re
import collections

from libs.configs import cfgs

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
        batch_size=batch_size,
        capacity=capacity,
        min_after_dequeue=min_after_dequeue)

def load_batch_data(lines, batch_size, withuin=True, ret_uin=False, pos_num=5, neg_num=50,max_channel_embed_size = 20,max_term_num = 50):
    feature_all, label, masked_channel, uin_list = [], [], [],[]

    def flatten_list(l):
        return [item for sublist in l for item in sublist]

    labels = []
    for line in lines:
        line_ele = line.decode().rstrip('\n').split('\t')
        if len(line_ele) < 2: continue

        feature_list_item = line_ele[1] if withuin else line_ele[0]
        feature_item = [ ' '.join(x.split('\001')) for x in feature_list_item.split('\002') ]

        feature_list = feature_item

        need_labels = [int(x) for x in line_ele[-1].split('\001')][:pos_num + neg_num]
        while len(need_labels) < pos_num + neg_num:
            need_labels += need_labels[pos_num:]
        need_labels = need_labels[:pos_num + neg_num]
        labels.append(need_labels)
        feature_info = [[0]*1, [0]*max_term_num, [0]*1, [0]*max_term_num, [0]*1, [0]*max_term_num, [0]*1, [0]*max_term_num,
                        [0]*1, [0]*max_term_num, [0]*1, [0]*max_term_num, [0]*1, [0]*max_term_num, [0]*1, [0]*max_term_num,
                        [0]*1, [0]*max_term_num, [0]*1, [0]*max_term_num, [0]*1, [0]*max_term_num, [0]*1, [0]*max_term_num,
                        [0]*1, [0]*max_term_num, [0]*1, [0]*max_term_num, [0]*1, [0]*max_term_num, [0]*1, [0]*max_term_num,
                        [0]*1, [0]*max_term_num, [0]*1, [0]*max_term_num,[0]*1, [0]*max_term_num, [0]*1, [0]*max_term_num]

        f_len = 0
        for index, feature in enumerate(feature_list):
            feature = feature.strip()
            if len(feature) == 0: continue
            if index == len(feature_info) - 1: break

            f_len += 1
            infos = re.split(' |\|', feature)
            index = 2 * (int(infos[0]))
            feature_info[index][0] = float(infos[0]) + 1  #[channel]
            index += 1
            for i, val in enumerate(infos[1:]):
                if len(val.strip()) == 0: continue
                feature_info[index][i] = float(val)
                if i == len(feature_info[index]) - 1: break

        feature_all.append(feature_info)
        uin_list.append(line_ele[0])
        masked_channel.append(int(line_ele[2]))

    if not feature_all: return {}
    feature_all = np.array(feature_all)
    terms = []
    for i in range(max_channel_embed_size):
        terms.append(feature_all[:, i * 2 + 1])
    term_ids = np.stack(terms, axis=1)
    feature_terms = np.reshape(
        np.array([flatten_list(x) for x in list(term_ids)]), [batch_size, max_channel_embed_size, max_term_num])

    feature_map = {
        "uin": np.expand_dims(np.array(uin_list), -1),
        "term_ids": feature_terms,
        "term_mask": np.ones([batch_size, max_channel_embed_size, max_term_num]),
        "masked_channel": np.reshape(np.array(masked_channel), [batch_size, 1]),
        "labels": np.array(labels)
    }
    return feature_map

def load_data_generator(lines, batch_size, withuin=False, generator=-1 ,pointwise = True,max_term_num = 50):
    feature_all, label, uin_list = [], [], []
    load_cnt = 0
    for line in lines:
        line_ele = line.decode().rstrip('\n').split('\t')
        if len(line_ele) < 2: continue

        feature_list_item = line_ele[1] if withuin else line_ele[0]
        feature_item = [ ' '.join(x.split('\001')) for x in feature_list_item.split('\002') ]

        feature_list = feature_item
        feature_info = [[0] * 1, [0] * max_term_num, [0] * 1, [0] * max_term_num, [0] * 1, [0] * max_term_num, [0] * 1,[0] * max_term_num,
                        [0] * 1, [0] * max_term_num, [0] * 1, [0] * max_term_num, [0] * 1, [0] * max_term_num, [0] * 1,[0] * max_term_num,
                        [0] * 1, [0] * max_term_num, [0] * 1, [0] * max_term_num, [0] * 1, [0] * max_term_num, [0] * 1,[0] * max_term_num,
                        [0] * 1, [0] * max_term_num, [0] * 1, [0] * max_term_num, [0] * 1, [0] * max_term_num, [0] * 1,[0] * max_term_num,
                        [0] * 1, [0] * max_term_num, [0] * 1, [0] * max_term_num, [0] * 1, [0] * max_term_num, [0] * 1,[0] * max_term_num,
                        [0] * 2]

        f_len = 0
        for index, feature in enumerate(feature_list):
            feature = feature.strip()
            if len(feature) == 0: continue
            if index == len(feature_info)-1: break

            f_len += 1
            infos = re.split(' |\|', feature)
            index = 2*(int(infos[0]))
            feature_info[index][0] = float(infos[0]) + 1
            index += 1
            for i, val in enumerate(infos[1:]):
                if len(val.strip()) == 0:continue
                feature_info[index][i] = float(val)
                if i == len(feature_info[index])-1: break

        feature_info[-1][0] = float(f_len)/len(feature_list)
        feature_info[-1][1] = float(len(feature_list)-f_len)/len(feature_list)


        if pointwise:
            for x in range(generator):
                feature_all.append(feature_info)
        else:
            feature_all.append(feature_info)

        load_cnt += 1
        if load_cnt == batch_size: break

    if not feature_all: return []

    return feature_all


def load_data(lines, batch_size, max_term_num, withuin=False, ret_uin=False):
    feature_all, label, uin_list, click_item = [], [], [],[]
    prelabel, neglabel = -1, -1
    preLolabel, negLolabel = -1, -1
    for line in lines:
        line_ele = line.decode().rstrip('\n').split('\t')
        if len(line_ele) < 2: continue

        feature_list_item = line_ele[1] if withuin else line_ele[0]


        feature_item = [ ' '.join(x.split('\001')) for x in feature_list_item.split('\002') ]

        if prelabel > 0 and random.random() > 0.2:
            neglabel = prelabel
            negLolabel = preLolabel
        else:
            neglabel = random.randint(1, cfgs.vocab_num)
            negLolabel = cfgs.test_domain


        torfL = int(line_ele[2])
        label_feature = line_ele[3].split('\001')
        torf = int( label_feature[random.randint(0, len(label_feature)-1)] )

        click_item.append([int(l) for l in label_feature])

        prelabel = torf
        preLolabel = torfL

        feature_list = feature_item
        feature_info = [[0] * 1, [0] * max_term_num, [0] * 1, [0] * max_term_num, [0] * 1, [0] * max_term_num, [0] * 1,[0] * max_term_num,
                        [0] * 1, [0] * max_term_num, [0] * 1, [0] * max_term_num, [0] * 1, [0] * max_term_num, [0] * 1,[0] * max_term_num,
                        [0] * 1, [0] * max_term_num, [0] * 1, [0] * max_term_num, [0] * 1, [0] * max_term_num, [0] * 1,[0] * max_term_num,
                        [0] * 1, [0] * max_term_num, [0] * 1, [0] * max_term_num, [0] * 1, [0] * max_term_num, [0] * 1,[0] * max_term_num,
                        [0] * 1, [0] * max_term_num, [0] * 1, [0] * max_term_num, [0] * 1, [0] * max_term_num, [0] * 1,[0] * max_term_num
                        ,[0]*2]

        f_len = 0
        for index, feature in enumerate(feature_list):
            feature = feature.strip()
            if len(feature) == 0: continue
            if index == len(feature_info)-1: break

            f_len += 1
            infos = re.split(' |\|', feature)
            index = 2*int(infos[0])
            feature_info[index][0] = float(infos[0]) + 1
            index += 1
            for i, val in enumerate(infos[1:]):
                if len(val.strip()) == 0:continue
                feature_info[index][i] = float(val)
                if i == len(feature_info[index])-1: break

        feature_info[-1][0] = float(f_len)/len(feature_list)
        feature_info[-1][1] = float(len(feature_list)-f_len)/len(feature_list)


        feature_all.append(feature_info)

        torf = float(torf)
        label.append([1.0, torf, torfL])

        feature_all.append(feature_info)
        label.append([0.0, float(neglabel), float(negLolabel)])

        if ret_uin and withuin:
            uin_list.append(line_ele[0])
            uin_list.append(line_ele[0])

        if len(label) == batch_size: break

    if not feature_all: return []

    if not ret_uin:
        return np.array(feature_all), np.array(label), click_item
    return np.array(feature_all), np.array(label), uin_list, click_item

def load_data_recall(lines, batch_size, max_term_num, pos_num=30, neg_num=30, withuin=False, ret_uin=False):
    feature_all, label, uin_list = [], [], []
    prelabel, neglabel = [], []
    preLolabel, negLolabel = [], []
    for line in lines:
        line_ele = line.decode().rstrip('\n').split('\t')
        if len(line_ele) < 2: continue

        feature_list_item = line_ele[1] if withuin else line_ele[0]


        feature_item = [ ' '.join(x.split('\001')) for x in feature_list_item.split('\002') ]

        torfL = int(line_ele[2])
        label_feature = line_ele[3].split('\001')
        torf = random.sample(label_feature, pos_num)

        neg_all_sample = list(set(range(1, cfgs.vocab_num + 1)) - set([int(i) for i in label_feature]))

        neglabel = random.sample(neg_all_sample,neg_num)


        feature_list = feature_item
        feature_info = [[0] * 1, [0] * max_term_num, [0] * 1, [0] * max_term_num, [0] * 1, [0] * max_term_num, [0] * 1,[0] * max_term_num,
                        [0] * 1, [0] * max_term_num, [0] * 1, [0] * max_term_num, [0] * 1, [0] * max_term_num, [0] * 1,[0] * max_term_num,
                        [0] * 1, [0] * max_term_num, [0] * 1, [0] * max_term_num, [0] * 1, [0] * max_term_num, [0] * 1,[0] * max_term_num,
                        [0] * 1, [0] * max_term_num, [0] * 1, [0] * max_term_num, [0] * 1, [0] * max_term_num, [0] * 1,[0] * max_term_num,
                        [0] * 1, [0] * max_term_num, [0] * 1, [0] * max_term_num, [0] * 1, [0] * max_term_num, [0] * 1,[0] * max_term_num
                        ,[0]*2]

        f_len = 0
        for index, feature in enumerate(feature_list):
            feature = feature.strip()
            if len(feature) == 0: continue
            if index == len(feature_info)-1: break

            f_len += 1
            infos = re.split(' |\|', feature)
            index = 2*int(infos[0])
            feature_info[index][0] = float(infos[0]) + 1
            index += 1
            for i, val in enumerate(infos[1:]):
                if len(val.strip()) == 0:continue
                feature_info[index][i] = float(val)
                if i == len(feature_info[index])-1: break

        feature_info[-1][0] = float(f_len)/len(feature_list)
        feature_info[-1][1] = float(len(feature_list)-f_len)/len(feature_list)


        feature_all.append(feature_info)
        torf = [float(t) for t in torf]
        neglabel = [float(t) for t in neglabel]
        if len(torf) != pos_num or len(neglabel) != neg_num:
            print("len(torf):",len(torf)," pos_num:",pos_num, " len(neglabel)",len(neglabel) ," neg_num:",neg_num)
            print("len(label_feature):", len(label_feature),"len(neg_all_sample)",len(neg_all_sample))
            exit()
        label.append([[1.0] * pos_num + [0.0] * neg_num, torf + neglabel, torfL])

        if ret_uin and withuin:
            uin_list.append(line_ele[0])
            uin_list.append(line_ele[0])

        if len(label) == batch_size: break

    if not feature_all: return []

    if not ret_uin:
        return np.array(feature_all), np.array(label)
    return np.array(feature_all), np.array(label), uin_list


def get_assignment_map_from_checkpoint(tvars, init_checkpoint):
    """Compute the union of the current variables and checkpoint variables."""
    initialized_variable_names = {}

    name_to_variable = collections.OrderedDict()
    for var in tvars:
        name = var.name
        m = re.match("^(.*):\\d+$", name)
        if m is not None:
            name = m.group(1)
        name_to_variable[name] = var

    init_vars = tf.train.list_variables(init_checkpoint)

    assignment_map = collections.OrderedDict()
    for x in init_vars:
        (name, var) = (x[0], x[1])
        if name not in name_to_variable:
            continue
        assignment_map[name] = name_to_variable[name]
        initialized_variable_names[name] = 1
        initialized_variable_names[name + ":0"] = 1

    return (assignment_map, initialized_variable_names)

def list_check(loss_list, train_auc_list, train_recall_list,test_auc_list, test_recall_list):
    if len(loss_list[0]) == 5: loss_list[0].pop(0)
    if len(train_auc_list[0]) == 5: train_auc_list[0].pop(0)
    if len(train_recall_list[0]) ==5: train_recall_list[0].pop(0)
    if len(test_auc_list[0]) == 5: test_auc_list[0].pop(0)
    if len(test_recall_list[0]) == 5: test_recall_list[0].pop(0)

def save_model_out(_sess, _model, _model_save_path, _save_flag):
    print('begin save model', file=sys.stderr)
    _model.saver.save(_sess, _model_save_path)
    print('save model done', file=sys.stderr)
    _save_flag = True
    return _save_flag

def get_file_list(file_path):
    if os.path.isfile(file_path):
        return [ file_path ]
    file_list = []
    for file_name in os.listdir( file_path ):
        if 'SUCCESS' in file_name: continue
        file_list.append( os.path.join(file_path, file_name) )
    return file_list

def cal_auc_out_1posNneg(auc_val, auc_sum, _writer, _step, _max_auc, _str='activc AUC'):
    if _writer is not None:
        _writer.add_summary(auc_sum, _step)
    _max_auc = max(_max_auc, auc_val)
    print('<'*20, file=sys.stderr)
    print('%d step, avg %s is %.4f. max test AUC is %.4f' % (_step, _str, auc_val, _max_auc), file=sys.stderr)
    return _max_auc

def cal_recall_out_1posNneg(recall_val, recall_sum, _writer, _step, _max_recall, _str='activc RECALL'):
    if _writer is not None:
        _writer.add_summary(recall_sum, _step)
    _max_recall = max(_max_recall, recall_val)
    print('<'*20, file=sys.stderr)
    print('%d step, avg %s is %.4f. max test recall is %.4f' % (_step, _str, recall_val, _max_recall), file=sys.stderr)
    print('<' * 20, file=sys.stderr)
    return _max_recall