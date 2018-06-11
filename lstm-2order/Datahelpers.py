import numpy as np
import nltk
import itertools

class Datahelper(object):
    def __init__(self, data_dir, limits):
        self.data_dir = data_dir
        self.train_data_path = data_dir + '/train_data.txt'
        self.dev_data_path = data_dir + '/dev_data.txt'
        self.test_data_path = data_dir + '/test.txt'
        self.train_feat_path = data_dir + '/train_data_feat.txt'
        self.dev_feat_path = data_dir + '/dev_data_feat.txt'
        self.test_feat_path = data_dir + '/test.feat.txt'
        self.limits = limits

        word_data, pos_data, tag_data = self.get_all_data(self.train_data_path)
        #self.word2idx, self.idx2word = self.get_dict(word_data)
        self.word2idx, self.emb_matrix = self.emb()
        self.pos2idx, self.idx2pos = self.get_dict(pos_data)
        self.feat2idx, self.feat_num = self.get_feat_dict()
        self.tag_1o_idx, self.tag_2o_idx = self.get_mul_dict(tag_data)
        self.idx_1o_tag = {self.tag_1o_idx[tag]: tag for tag in self.tag_1o_idx}
        self.idx_2o_tag = {self.tag_2o_idx[tag]: tag for tag in self.tag_2o_idx}

        self.train_set = self.get_data(self.train_data_path, self.train_feat_path, self.word2idx, self.pos2idx, self.tag_1o_idx, self.tag_2o_idx, self.feat2idx)
        self.dev_set = self.get_data(self.dev_data_path, self.dev_feat_path, self.word2idx, self.pos2idx, self.tag_1o_idx, self.tag_2o_idx, self.feat2idx)
        self.test_set = self.get_data(self.test_data_path, self.test_feat_path, self.word2idx, self.pos2idx, self.tag_1o_idx, self.tag_2o_idx, self.feat2idx)

    def get_all_data(self, train_data_path):
        data = open(train_data_path, 'r').read().strip().split('\n\n')
        word_data = [[word.split(' ')[0].lower() for word in sentence.split('\n')] for sentence in data]
        pos_data = [[word.split(' ')[1] for word in sentence.split('\n')] for sentence in data]
        #bio_data = [[word.split(' ')[2] for word in sentence.split('\n')] for sentence in data]
        tag_data = [[word.split(' ')[2] for word in sentence.split('\n')] for sentence in data]
        return word_data, pos_data, tag_data

    def get_mul_dict(self, data):

        tag_1o_idx = {'I-SBAR': 12, 'B-SBAR': 6, 'B-ADJP': 8, 'I-VP': 5, 'UNKNOWN': 23, 'I-PP': 13, 'B-INTJ': 16, 'I-NP': 3,
        'B-VP': 4, 'I-ADVP': 10, 'B-UCP': 21, 'I-UCP': 22, 'I-ADJP': 11, 'B-PRT': 14, 'I-CONJP': 19, 'O': 7,
        'B-LST': 15, 'B-CONJP': 18, 'B-ADVP': 9, 'I-PRT': 20, 'B-PP': 2, 'I-INTJ': 17, 'PADDING': 0, 'B-NP': 1}

        idx_2o = 1
        tag_2o_idx = {}
        tag_2o_idx[(0,0)] = 0

        data_1o = [[tag_1o_idx[tag] for tag in tags] for tags in data]
        data_2o = [[(([0] + tags)[i - 1], ([0] + tags)[i]) for i in range(1, len([0] + tags))] for tags in data_1o]

        for sent_2o_tag in data_2o:
            for w_tag in sent_2o_tag:
                if w_tag not in tag_2o_idx:
                    tag_2o_idx[w_tag] = idx_2o
                    idx_2o += 1

        tag_1o_idx['UNKNOWN'] = len(tag_1o_idx)
        tag_2o_idx['UNKNOWN'] = len(tag_2o_idx)

        return tag_1o_idx, tag_2o_idx

    def get_dict(self, data):
        word_freq = nltk.FreqDist(itertools.chain(*data))
        data = sorted(word_freq.items(), key=lambda x: (x[1], x[0]), reverse=True)[:5000 - 2]
        key_set = set()
        for pair in data:
            key_set.add(pair[0])
        word2idx = {w:i+1 for i,w in enumerate(key_set)}
        word2idx['UNKNOWN'] = len(word2idx) + 1
        word2idx['PADDING'] = 0
        idx2word = {word2idx[w]:w for w in word2idx}
        return word2idx, idx2word

    def get_feat_dict(self):
        feat2idx = {'PADDING': 0, 'UNKNOWN': 1}
        idx = 2
        with open(self.data_dir + "/train.feat.txt") as f:
            data = [[[w for w in row.strip().split(' ')] for row in sent.strip().split('\n')] for sent in
                    f.read().strip().split('\n\n')]
            feat_num = len(data[0][0]) - 1
            for sent in data:
                for row in sent:
                    for w in row[1:]:
                        if w not in feat2idx:
                            feat2idx[w] = idx
                            idx += 1
        return feat2idx, feat_num

    def get_data(self, path, feat_path, word_dict, pos_dict,  tag_1o_dict, tag_2o_dict, feat_dict):
        data = open(path, 'r').read().strip().split('\n\n')
        feat_data = open(feat_path, 'r').read().strip().split('\n\n')
        if self.limits > 0:
            data = data[:self.limits]

        word_data = [[word.split(' ')[0].lower() for word in sentence.split('\n')] for sentence in data]
        pos_data = [[word.split(' ')[1] for word in sentence.split('\n')] for sentence in data]
        #bio_data = [[word.split(' ')[2] for word in sentence.split('\n')] for sentence in data]
        tag_1o_data = [[word.split(' ')[2] for word in sentence.split('\n')] for sentence in data]
        word_data = [[word_dict[w] if w in word_dict else word_dict['UNKNOWN'] for w in sentence] for sentence in word_data]
        pos_data = [[pos_dict[w] if w in pos_dict else pos_dict['UNKNOWN'] for w in sentence] for sentence in pos_data]
        #bio_data = [[bio_dict[w] if w in bio_dict else bio_dict['UNKNOWN'] for w in sentence] for sentence in bio_data]
        tag_1o_data = [[tag_1o_dict[w] if w in tag_1o_dict else tag_1o_dict['UNKNOWN'] for w in sentence] for sentence
                       in tag_1o_data]

        tags_2o = [[(([0] + tag)[i - 1], ([0] + tag)[i]) for i in range(1, len([0] + tag))] for tag in tag_1o_data]
        tag_2o_data = [[tag_2o_dict[w] if w in tag_2o_dict else tag_2o_dict['UNKNOWN'] for w in sentence] for sentence
                       in tags_2o]

        feat_data = [[[w for w in row.strip().split(' ')[1:]]for row in sentence.strip().split('\n')] for sentence in feat_data]
        feat_data = [[[feat_dict[w] if w in feat_dict else feat_dict['UNKNOWN'] for w in row]for row in sentence]for sentence in feat_data]
        print ("feat_size = ", len(feat_data))
        return word_data, pos_data, tag_2o_data, feat_data

    def batch_iter(self, data, batch_size, shuffle, feat_num):
        word_data, pos_data, tag_data, feat_data = data
        data_size = len(word_data)
        print ("datasize = ", data_size)
        num_batches_per_epoch = int((data_size - 1) / batch_size) + 1
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            word_data = np.array(word_data)[shuffle_indices]
            pos_data = np.array(pos_data)[shuffle_indices]
            #bio_data = np.array(bio_data)[shuffle_indices]
            tag_data = np.array(tag_data)[shuffle_indices]
            feat_data = np.array(feat_data)[shuffle_indices]
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            max_sent_len = max([len(sample) for sample in word_data[start_index: end_index]])
            batch_data = {'x_word':[], 'x_pos':[], 'x_len':[],'x_feat':[], 'y_tag':[]}
            for sent_word, sent_pos, sent_tag, sent_feat in zip(word_data[start_index:end_index], pos_data[start_index:end_index],
                                                                tag_data[start_index:end_index], feat_data[start_index:end_index]):
                sent_len = len(sent_word)
                sent_word = sent_word + [0] * (max_sent_len - sent_len)
                sent_pos = sent_pos + [0] * (max_sent_len - sent_len)
                #sent_bio = sent_bio + [0] * (max_sent_len - sent_len)
                sent_tag = sent_tag + [0] * (max_sent_len - sent_len)
                feat_pad = [[0] * feat_num]
                sent_feat = sent_feat + feat_pad * (max_sent_len - sent_len)
                batch_data['x_word'].append(sent_word)
                batch_data['x_pos'].append(sent_pos)
                #batch_data['x_bio'].append(sent_bio)
                batch_data['x_len'].append(sent_len)
                batch_data['y_tag'].append(sent_tag)
                batch_data['x_feat'].append(sent_feat)
            yield batch_data

    def emb(self):
        word2idx = {'PADDING':0, 'UNKNOWN':1}
        idx2word = ['PADDING', 'UNKNOWN']
        vec = [np.zeros(50),np.random.uniform(-0.5, 0.5, 50)]
        with open('/home/zhangyi/chunking/data'+ '/senna_2.txt') as f:
            f_vec = [emb for emb in f.read().strip().split('\n')]
        for i in range(len(f_vec)):
            tem = f_vec[i].strip().split(' ')
            word = tem[0]
            #print tem
            if word != 'PADDING' and word !='UNKNOWN':
                v = np.array([float(num) for num in tem[1:]])
                word2idx[word] = len(idx2word)
                idx2word.append(word)
                vec.append(v)
        print (len(word2idx), len(vec))
        return word2idx, np.array(vec,dtype = np.float32)

