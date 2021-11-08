
#-----------------------------------------------------------------------------Train configs
loss_step = 40  # Print Train AUC+Loss for every loss_step
test_step = 40  # Print Test AUC for every test_step
save_step = 100

G_steps = 1
D_steps = 5

batch_size = 32

learning_rate = 0.0001

domain_cnt = 20

test_domain = 3

pos_num = 10
neg_num = 50

max_term_num = 50
max_channel_embed_size = 20

tag_size = 479509
generate_sample_num = 1000
d_pos_num = 10
reward_num = 50    #select neg sample num

channel_num = 20

vocab_num = 17771

channels = [i for i in range(channel_num)]
tag_list = [[0, vocab_num]] * channel_num

test_pos_num = 10
test_neg_num = 1000

label_file = './data/netflix_pub/recall_test_item/label%d' % test_domain

pretrain_iter_g = 5000
pretrain_iter_d = 20000

#-------------------------------------------------------------------------------------------Model params
vocab_size = vocab_num + 1
hidden_size = 64
ffn_size = 128
num_attention_head_g = 8
num_hidden_layers_g = 4

attention_drop_prob_g = 0.1
hidden_drop_prob_g = 0.1
initializer_range_g = 0.02

num_attention_head_d = 16
num_hidden_layers_d = 1

attention_drop_prob_d = 0.1
hidden_drop_prob_d = 0.1
initializer_range_d = 0.1

cnn_size = 16

alpha = 0.0

#-----------------------------------------------------------------------------------path_param
prefix_dir = './data/netflix_pub/split_60'

# predict tag tmp file
predict_data_save_path = './output/tag_predict'

save_model_dir = './output/trained_model/whole_model'
OUTPUT_PATH = './output'
log_dir = OUTPUT_PATH + '/log'
sum_dir = OUTPUT_PATH + '/tensorbord'

# train file
train_data_file_list = [prefix_dir + '/%d/train' % (i) for i in channels]
# test file
test_data_file_list = [prefix_dir + '/%d/test' % (i) for i in channels]
