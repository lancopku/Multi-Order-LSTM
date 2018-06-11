import tensorflow as tf
class Bi_lstm(object):
    def __init__(self, batch_size, vocab_size, pos_size, word_emb_size, pos_emb_size,
                 hidden_dim, tag_size, emb_matrix, use_emb, dropout_keep_prob,
                 feat_size, feat_num, feat_dim):
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.word_emb_size = word_emb_size
        self.pos_emb_size = pos_emb_size
        self.hidden_dim = hidden_dim
        self.tag_size = tag_size
        self.use_emb = use_emb

        self.dropout_keep_prob = tf.placeholder(tf.float32)
        self.x_word = tf.placeholder(tf.int32, [None, None])
        self.x_pos = tf.placeholder(tf.int32, [None, None])
        self.x_len = tf.placeholder(tf.int32, [None])
        self.x_feat = tf.placeholder(tf.int32,[None, None, feat_num])
        self.y = tf.placeholder(tf.int32, [None, None])
        length = tf.shape(self.x_word)[1]

        with tf.name_scope("embedding"):
            #if use_emb == True:
            self.word_emb = tf.Variable(emb_matrix, trainable = True)
            #else:
            #self.word_emb = tf.Variable(tf.random_uniform([vocab_size, word_emb_size], -0.5, 0.5))
            self.pos_emb = tf.Variable(tf.random_uniform([pos_size, pos_emb_size], -0.5, 0.5))
            self.feature_emb = tf.Variable(tf.random_uniform([feat_size, feat_dim], -0.5, 0.5))
            x_1 = tf.nn.embedding_lookup(self.word_emb, self.x_word)

            x_2 = tf.nn.embedding_lookup(self.pos_emb, self.x_pos)
            x_ft = tf.nn.embedding_lookup(self.feature_emb, self.x_feat)
            x_ft = tf.reshape(x_ft, [-1, length, feat_num * feat_dim])
            self.x = tf.concat([x_1, x_2, x_ft], axis=2)
            #self.x = x_1
            self.x_drop = tf.nn.dropout(self.x, self.dropout_keep_prob)
            #self.x_drop = tf.concat([x_1_drop, x_2_drop, x_3_drop, x_ft_drop], axis = 2)

        with tf.name_scope("cell"):
            cell = tf.contrib.rnn.BasicLSTMCell(num_units = hidden_dim)
            outputs, states  = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=cell,
                cell_bw=cell,
                dtype=tf.float32,
                sequence_length=self.x_len,
                inputs=self.x_drop)
            output_fw, output_bw = outputs
            states_fw, states_bw = states

        with tf.name_scope("outputs"):

            self.lstm_outputs = output_fw + output_bw
            # Output layer weights
            W = tf.get_variable(
                name="W",
                initializer=tf.random_uniform_initializer(),
                shape=[self.hidden_dim, self.tag_size])
            self.lstm_outputs_flat = tf.reshape(self.lstm_outputs, [-1, hidden_dim])
            self.h_drop = tf.nn.dropout(self.lstm_outputs_flat, 1.0)
            logits_flat = tf.matmul(self.h_drop, W)

            #logits_flat = tf.matmul(self.lstm_outputs_flat, W)
            self.probs_flat = tf.nn.softmax(logits_flat)
            self.score = tf.reshape(logits_flat, [-1, length, tag_size] )
            self.prob = tf.reshape(self.probs_flat, [-1, length, tag_size])

        with tf.name_scope("loss"):
            y_flat =  tf.reshape(self.y, [-1])
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = logits_flat, labels = y_flat)
            # Mask the losses
            mask = tf.sign(tf.to_float(y_flat))
            masked_losses = mask * losses
            # Bring back to [B, T] shape
            masked_losses = tf.reshape(masked_losses,  tf.shape(self.y))
            # Calculate mean loss
            mean_loss_by_example = tf.reduce_sum(masked_losses, axis=1) / tf.to_float(self.x_len)
            self.mean_loss = tf.reduce_mean(mean_loss_by_example)
            self.mean_loss = tf.reduce_mean(masked_losses)

        with tf.name_scope("predict_tag"):
            prediction_y = tf.argmax(self.probs_flat, 1)
            self.prediction_y = tf.reshape(prediction_y, tf.shape(self.x_word))


        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.mean_loss, tvars), 5.0)
        optimizer = tf.train.AdamOptimizer(0.001)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))


    def __call__(self, sess, x):
        loss, _ = sess.run([self.mean_loss, self.train_op],
                           {self.x_word: x['x_word'], self.x_pos: x['x_pos'],
                             self.x_len: x['x_len'], self.x_feat:x['x_feat'],
                             self.y: x['y_tag'], self.dropout_keep_prob:0.5})
        return loss

    def generate(self, sess, x):
        predictions, prob, score = sess.run([self.prediction_y, self.prob, self.score],
                              {self.x_word: x['x_word'], self.x_pos: x['x_pos'],
                               self.x_len: x['x_len'], self.x_feat:x['x_feat'],
                               self.dropout_keep_prob: 1.0})
        pred = []
        for i,sent_pred in enumerate(predictions):
            pred.append(sent_pred[:x['x_len'][i]])
        return pred, prob

    def save(self, sess, path):
        saver = tf.train.Saver(tf.global_variables())
        saver.save(sess, path + "checkpoint.data")

    def load(self, sess, path):
        saver = tf.train.Saver(tf.global_variables())
        saver.restore(sess, path + "checkpoint.data")