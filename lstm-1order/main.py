import tensorflow as tf
import numpy as np
import os
import time
from model import Bi_lstm
from Datahelpers import Datahelper
import datetime
from functools import cmp_to_key
#from ulit import eval_ner
tf.app.flags.DEFINE_integer("batch_size", 1, "Batch size of train set.")
tf.app.flags.DEFINE_string("cell", "lstm", "Rnn cell.")
tf.app.flags.DEFINE_integer("max_epoch", 10, "Number of training epoch.")
tf.app.flags.DEFINE_integer("hidden_size", 300, "Size of each layer.")
tf.app.flags.DEFINE_integer("word_emb_size", 50, "Size of embedding.")
tf.app.flags.DEFINE_integer("pos_emb_size", 20, "Size of embedding.")
tf.app.flags.DEFINE_integer("bio_emb_size", 20, "Size of embedding.")
tf.app.flags.DEFINE_integer("limits", 0,'max data set size')
tf.app.flags.DEFINE_string("gpu", '0', "GPU id.")
tf.app.flags.DEFINE_string("dir",'../data','data set directory')
tf.app.flags.DEFINE_string("mode",'test','train or test')
tf.app.flags.DEFINE_integer("report", 2000,'report')
tf.app.flags.DEFINE_string("save",'1_kjflas','save directory')
tf.app.flags.DEFINE_string("word_emb",'sskip','pretrained_embedding')
tf.app.flags.DEFINE_string("dropout_keep_prob",1.0,'dropout keep probablity')
tf.app.flags.DEFINE_string("rnn_save",'rnn_saved','save directory')
tf.app.flags.DEFINE_string("pretrained",'False','save directory')
tf.app.flags.DEFINE_string("use_emb",'False','whether use pretrained embeddings')
FLAGS = tf.app.flags.FLAGS

log_file = FLAGS.dir + '/log_emb_True.txt'
with open(FLAGS.dir +"/dev_data.txt") as infile:
    gold_dev = [[w for w in sent.strip().split('\n')]for sent in infile.read().split('\n\n')]

with open(FLAGS.dir +"/test.txt") as infile_test:
    gold_test = [[w for w in sent.strip().split('\n')]for sent in infile_test.read().split('\n\n')]

def write_log(s):
    with open(log_file, 'a') as f:
        f.write(s)

class Topk_list(object):
    def __init__(self, k):
        self.k = k
        self.data = []
    def Push(self, idx, elem):
        if len(self.data) < self.k:
            self.data.append([idx, elem])
            if len(self.data) == self.k:
                #self.data.sort(lambda x,y : cmp(x[1],y[1]))
                self.data.sort(key = lambda x: x[1])
        else:
            if self.data[0][1] < elem:
                self.data[0] = [idx, elem]
                #self.data.sort(lambda x,y : cmp(x[1],y[1]))
                self.data.sort(key = lambda x: x[1])
    def top_k(self):
        topk = []
        for num in self.data:
            topk.append(num[0])
        return topk


def train(sess, datahelper, model):
    if FLAGS.pretrained == 'True':
        model.load(sess,save_dir)
        evaluate(sess, datahelper, model, gold_dev)

    write_log("##############################\n")
    for flag in FLAGS.__flags:
        write_log(flag + " = " + str(FLAGS.__flags[flag]) + '\n')
    write_log("##############################\n")
    train_set = datahelper.train_set
    #print train_set[4][:3]
    global_step = 0
    best_result = 93.8
    for _ in range(FLAGS.max_epoch):
        loss, start_time = 0.0, time.time()
        for x in datahelper.batch_iter(train_set, FLAGS.batch_size, True, datahelper.feat_num):
            loss += model(sess, x)
            global_step += 1
            if (global_step % FLAGS.report == 0):
                cost_time = time.time() - start_time
                write_log("%d : loss = %.3f, time = %.3f \n" % (global_step // FLAGS.report, loss, cost_time))
                print ("%d : loss = %.3f, time = %.3f " % (global_step // FLAGS.report, loss, cost_time))
                loss, start_time = 0.0, time.time()
                result_dev = evaluate(sess, datahelper, model, 'dev')
                result_test = evaluate(sess, datahelper, model, 'test')
                F1_dev,  overall_result_dev = result_dev

                if F1_dev > best_result:
                    print ("saving model......")
                    cur_save_dir = '../file_1o/' + 'dev_' + str(F1_dev)'/'
                    os.mkdir(cur_save_dir)
                    model.save(sess, cur_save_dir)
                    print ("model with " + 'dev result ' + str(F1_dev) + ' saved')
                    best_result = F1_dev

def test(sess, datahelper, model):
    model.load(sess, '../file_1o/saved_1o_model/')
    print ("save_dir = ",save_dir)
    evaluate(sess, datahelper, model, 'dev')
    evaluate(sess,datahelper, model, 'test')

def evaluate(sess, datahelper, model, data_type):
    if data_type == 'dev':
        test_set = datahelper.dev_set
        outfile_name = 'outfile_dev.txt'
        gold = gold_dev
    elif data_type == 'test':
        test_set = datahelper.test_set
        outfile_name = 'outfile_test.txt'
        gold = gold_test

    pred = []
    time_str = datetime.datetime.now().isoformat()
    print("{}".format(time_str))
    outfile = open("../file_1o/prob_1o_%s.txt" % data_type, 'w')
    tagfile = open("../file_1o/idx_1o_%s.txt" % data_type, 'w')
    pred = []
    time_str = datetime.datetime.now().isoformat()
    print("{}".format(time_str))
    test_start = datetime.datetime.now()
    for x in datahelper.batch_iter(test_set, FLAGS.batch_size, False, datahelper.feat_num):
        predictions, prob = model.generate(sess, x)
        pred.extend(predictions)
        
        for num, prob_sent in enumerate(prob):
            for prob_word in prob_sent[:x['x_len'][num]]:
                top = Topk_list(5)
                for tag_idx, w in enumerate(prob_word):
                    top.Push(tag_idx, w)
                    outfile.write('%s' %(w) + ' ')
                top_idx = top.top_k()
                for idx in top_idx:
                    tagfile.write(str(idx)+ ' ')
                tagfile.write('\n')
                outfile.write('\n')
            tagfile.write('\n')
            outfile.write('\n')
        
    test_end = datetime.datetime.now()
    test_time = test_end - test_start
    print ("test - time = ", test_time)
   
    with open(outfile_name, 'w') as f:
        for test_sent, pred_sent in zip(gold, pred):
            pre_tag = 'NULL'
            for test_line, pred_line in zip(test_sent, pred_sent):
                test_line = test_line.strip().split()
                sp_pretag = pre_tag.split('-')
                cur_tag = datahelper.idx_1o_tag[pred_line]
                sp_curtag = cur_tag.split('-')

                if len(sp_pretag) == 2 and len(sp_curtag) == 2:
                    pre_chunk = sp_pretag[0]
                    pre_type = sp_pretag[1]
                    cur_chunk = sp_curtag[0]
                    cur_type = sp_curtag[1]
                    if pre_chunk == 'B' and cur_chunk == 'I' and pre_type != cur_type:
                        cur_tag = cur_chunk + '-' + pre_type

                pre_tag = datahelper.idx_1o_tag[pred_line]
                test_line.append(cur_tag)
                f.write('{}\n'.format(" ".join(test_line)))
            f.write("\n")
        exe_command = 'perl conlleval < %s' % outfile_name
    result = os.popen(exe_command).readlines()
    for line in result:
        write_log(line)
    F1 = (result[1].split('  '))[-1]
    print ("%s f1 = %s" % (data_type, F1))
    return float(F1), result[1]

def main():
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        datahelper = Datahelper(FLAGS.dir, FLAGS.limits)
        vocab_size = len(datahelper.word2idx) + 2
        pos_size = len(datahelper.pos2idx)
        tag_size = len(datahelper.tag_1o_idx)
        print (datahelper.tag_1o_idx)
        print ("len = ", len(datahelper.idx_1o_tag))
        emb_matrix = datahelper.emb_matrix

        feat_size = len(datahelper.feat2idx)
        model = Bi_lstm(FLAGS.batch_size, vocab_size, pos_size,  FLAGS.word_emb_size,
                        FLAGS.pos_emb_size, FLAGS.hidden_size, tag_size, emb_matrix,
                        FLAGS.use_emb, FLAGS.dropout_keep_prob, feat_size, datahelper.feat_num, 10)
        sess.run(tf.global_variables_initializer())
        if FLAGS.mode == 'train':
            train(sess, datahelper, model)
        if FLAGS.mode == 'test':
            test(sess, datahelper, model)

if __name__ == '__main__':
    with tf.device('/gpu:' + FLAGS.gpu):
        tf.set_random_seed(6)
        main()
