# -*- coding: utf-8 -*-
import os
import sys
import time
import numpy as np
import tensorflow as tf
from tensorflow.contrib.crf import crf_log_likelihood
from tensorflow.contrib.crf import viterbi_decode
from tensorflow.contrib.rnn import LSTMCell
from tensorflow.contrib.layers.python.layers import initializers
from fenci.embedding import Data

os.environ["TF_CPP_MIN_LOG_LEVEL"]='3'
class BiLSTM_CRF(object):
    def __init__(self, args, embeddings, tag2label, word2id, tf_config):
        self.args = args
        self.embeddings = embeddings
        self.word2id = word2id
        self.tag2label = tag2label
        self.tf_config = tf_config

        self.model_path = args.model_path
        self.ckpt_prefix = args.ckpt_prefix
        self.batch_size = args.batch_size
        self.epochs = args.epoch
        self.hidden_dim = args.hidden_dim
        self.optimizer = args.optimizer
        self.learning_rate = args.learning_rate
        self.clip = args.clip
        self.dropout_keep_prob = args.dropout_keep_prob
        self.num_tags = len(tag2label) // 2

        self.CRF = args.CRF
        self.shuffle = args.shuffle
        self.update_embedding = args.update_embedding
        self.evaluate_mode = 2

        # self.logger = get_logger("model_train_log")
        self.result_path = "result"
        self.dd = Data(args)

    # 建图
    def build_graph(self):
        self.add_placeholders()
        self.lookup_layer_op()
        self.biLSTM_layer_op()
        self.softmax_pred_op()
        self.loss_op()
        self.trainstep_op()
        self.init_op()

    # 添加占位符
    def add_placeholders(self):
        # 输入占位符
        self.word_ids = tf.placeholder(tf.int32, shape=[None, None], name="word_ids")
        # 标签占位符
        self.labels = tf.placeholder(tf.int32, shape=[None, None], name="labels")
        # 序列长度占位符
        self.sequence_lengths = tf.placeholder(tf.int32, shape=[None], name="sequence_lengths")
        # dropout占位符
        self.dropout_pl = tf.placeholder(dtype=tf.float32, shape=[], name="dropout")
        # 学习率占位符
        self.learning_rate_pl = tf.placeholder(dtype=tf.float32, shape=[], name="lr")

    # 创建嵌入层
    def lookup_layer_op(self):
        with tf.variable_scope("words"):
            # 初始化词向量embedding矩阵
            _word_embeddings = tf.Variable(self.embeddings,
                                           dtype=tf.float32,
                                           trainable=self.update_embedding,
                                           name="_word_embeddings")
            # 查询单词的向量表达获得inputs （告诉TensorFlow在哪里去查找我们的词汇id）
            word_embeddings = tf.nn.embedding_lookup(params=_word_embeddings,
                                                     ids=self.word_ids,
                                                     name="word_embeddings")

        self.word_embeddings = tf.nn.dropout(word_embeddings, self.dropout_pl)

    # 创建向前 LSTM 网络和向后 LSTM 网络
    def biLSTM_layer_op(self):
        with tf.variable_scope("bi-lstm"):
            # 前向和后向lstm
            cell_fw = LSTMCell(num_units=self.hidden_dim, forget_bias=1.0, state_is_tuple=True)
            cell_bw = LSTMCell(num_units=self.hidden_dim, forget_bias=1.0, state_is_tuple=True)
            (output_fw_seq, output_bw_seq), _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_fw,
                                                                                cell_bw=cell_bw,
                                                                                inputs=self.word_embeddings,
                                                                                sequence_length=self.sequence_lengths,
                                                                                dtype=tf.float32)
            # 连接前向，后向lstm的结果并进行dropout
            output = tf.concat([output_fw_seq, output_bw_seq], axis=-1)
            output = tf.nn.dropout(output, self.dropout_pl)

            # 隐含层，并预测
            with tf.variable_scope("full-connect"):
                W = tf.get_variable(name="weights",
                                    dtype=tf.float32,
                                    shape=[2 * self.hidden_dim, self.num_tags],
                                    initializer=tf.contrib.layers.xavier_initializer(),
                                    regularizer=tf.contrib.layers.l2_regularizer(0.001))

                b = tf.get_variable(name="bias",
                                    dtype=tf.float32,
                                    shape=[self.num_tags],
                                    initializer=tf.zeros_initializer())

                s = tf.shape(output)
                output = tf.reshape(output, [-1, 2 * self.hidden_dim])
                pred = tf.matmul(output, W) + b

                self.logits = tf.reshape(pred, [-1, s[1], self.num_tags])

    # loss
    def loss_op(self):
        if self.CRF:
            self.transition_params = tf.get_variable(
                "transitions",
                shape=[self.num_tags, self.num_tags],
                initializer=initializers.xavier_initializer())

            # 计算crf损失，最大似然估计
            log_likelihood, self.transition_params = crf_log_likelihood(inputs=self.logits,
                                                                        tag_indices=self.labels,
                                                                        transition_params=self.transition_params,
                                                                        sequence_lengths=self.sequence_lengths)
            self.loss = -tf.reduce_mean(log_likelihood)
        else:
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits,
                                                                    labels=self.labels)

            # 保留跟句子实际长度相同的张量
            mask = tf.sequence_mask(self.sequence_lengths)
            self.losses = tf.boolean_mask(losses, mask)
            self.loss = tf.reduce_mean(self.losses)

        tf.summary.scalar("loss", self.loss)


    # softmax层
    def softmax_pred_op(self):
        if not self.CRF:
            self.labels_softmax_ = tf.argmax(self.logits, axis=-1)
            self.labels_softmax_ = tf.cast(self.labels_softmax_, tf.int32)

    # 激活层
    def trainstep_op(self):
        # 定义激活函数
        with tf.variable_scope("train_step"):
            self.global_step = tf.Variable(0, name="global_step", trainable=False)
            if self.optimizer == 'Adam':
                optim = tf.compat.v1.train.AdamOptimizer(learning_rate=self.learning_rate_pl)
            elif self.optimizer == 'Adadelta':
                optim = tf.compat.v1.train.AdadeltaOptimizer(learning_rate=self.learning_rate_pl)
            elif self.optimizer == 'Adagrad':
                optim = tf.compat.v1.train.AdagradOptimizer(learning_rate=self.learning_rate_pl)
            elif self.optimizer == 'RMSProp':
                optim = tf.compat.v1.train.RMSPropOptimizer(learning_rate=self.learning_rate_pl)
            elif self.optimizer == 'Momentum':
                optim = tf.compat.v1.train.MomentumOptimizer(learning_rate=self.learning_rate_pl)
            elif self.optimizer == 'SGD':
                optim = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=self.learning_rate_pl)
            else:
                optim = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=self.learning_rate_pl)

            grads_and_vars = optim.compute_gradients(self.loss)
            grads_and_vars_clip = [[tf.clip_by_value(g, -self.clip, self.clip), v] for g, v in grads_and_vars]
            self.train_op = optim.apply_gradients(grads_and_vars_clip, global_step=self.global_step)

    # tensorflow初始化
    def init_op(self):
        self.init_op = tf.compat.v1.global_variables_initializer()

    # 保存训练过程
    def add_summary(self, sess):
        self.merged = tf.compat.v1.summary.merge_all()  # 合并默认图中收集的所有摘要
        self.file_writer = tf.compat.v1.summary.FileWriter("model_summaries", sess.graph)

    # 训练模型
    def train(self, train, dev):
        saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables())
        with tf.Session(config=self.tf_config) as sess:
            sess.run(self.init_op)
            self.add_summary(sess)
            for epoch in range(self.epochs):
                self.run_one_epoch(sess, train, dev, epoch, saver)

    def run_one_epoch(self, sess, train, dev, epoch, saver):
        # 总batch_size个数
        num_batches = (len(train) + self.batch_size - 1) // self.batch_size
        start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

        batches = self.dd.batch_yield(train, self.batch_size, self.word2id, self.tag2label, shuffle=self.shuffle)
        for step, (seqs, labels) in enumerate(batches):
            sys.stdout.write(' processing: {} batch / {} batches.'.format(step + 1, num_batches) + '\r')
            step_num = epoch * num_batches + step + 1
            feed_dict, _ = self.get_feed_dict(seqs, labels, self.learning_rate, self.dropout_keep_prob)

            _, loss_train, logits_train, sequence_lengths_train, transition_params, summary, step_num_ = sess.run([self.train_op, self.loss,
                                                                                                self.logits, self.sequence_lengths,
                                                                                                self.transition_params,
                                                                                                self.merged, self.global_step],
                                                                                               feed_dict=feed_dict)

            ####################################################################################################################################
            # # 训练过程中的准确率
            # correct_seq = 0
            # total_num = 0
            # for logit, seq_len, label in zip(logits_train, sequence_lengths_train, labels):
            #     viterbi_seq, _ = viterbi_decode(logit[:seq_len], transition_params)
            #     for v, l in zip(viterbi_seq, label):
            #         correct_seq += np.sum((np.equal(v, l)))
            #     total_num += seq_len
            # acc = correct_seq / total_num
            ####################################################################################################################################
            if step + 1 == 1 or (step + 1) % 20 == 0 or step + 1 == num_batches:
                print('{} epoch {}, step {}, loss: {:.4}, global_step: {}' \
                      .format(start_time, epoch + 1, step + 1,loss_train, step_num))

            self.file_writer.add_summary(summary, step_num)

            if step + 1 == num_batches:
                saver.save(sess, self.ckpt_prefix, global_step=step_num)

        print('===============================validation / test===============================')
        correct_seq_test = 0
        total_num_test = 0
        batches_test = self.dd.batch_yield(dev, self.batch_size, self.word2id, self.tag2label, shuffle=self.shuffle)
        for (sent_, label_) in batches_test:
            feed_dict, seq_len_list = self.get_feed_dict(sent_, label_, dropout_keep_prob=1.0)
            if self.CRF:
                logits, transition_params = sess.run([self.logits, self.transition_params],
                                                      feed_dict=feed_dict)
                label_pre = []
                for logit, seq_len in zip(logits, seq_len_list):
                    viterbi_seq, _ = viterbi_decode(logit[:seq_len], transition_params)
                    label_pre.append(viterbi_seq)
            else:
                label_pre = sess.run(self.labels_softmax_, feed_dict=feed_dict)

            for v, l in zip(label_pre, label_):
                correct_seq_test += np.sum((np.equal(v, l)))
            total_num_test += np.sum(seq_len_list)
        acc_test = correct_seq_test / total_num_test
        print("-----------test_acc:{}------------".format(acc_test))
        # label_list_dev, seq_len_list_dev = self.dev_one_epoch(sess, dev)
        # self.evaluate(label_list_dev, seq_len_list_dev, dev, epoch)

    # 生成sess.run中feed_dict参数，参数包括：句子id，句子真实长度，学习率，dropout_keep_rate
    def get_feed_dict(self, seqs, labels=None, lr=None, dropout_keep_prob=None):
        # 句子ID，句子真实长度
        word_ids, seq_len_list = self.dd.pad_sequences(seqs, pad_mark=0)

        feed_dict = {self.word_ids: word_ids,
                     self.sequence_lengths: seq_len_list}
        if labels is not None:
            labels_, _ = self.dd.pad_sequences(labels, pad_mark=0)
            feed_dict[self.labels] = labels_
        if lr is not None:
            feed_dict[self.learning_rate_pl] = lr
        if dropout_keep_prob is not None:
            feed_dict[self.dropout_pl] = dropout_keep_prob

        return feed_dict, seq_len_list

    def dev_one_epoch(self, sess, dev):
        """
        :param sess:
        :param dev:
        :return:
        """
        label_list, seq_len_list = [], []
        for seqs, labels in self.dd.batch_yield(dev, self.batch_size, self.word2id, self.tag2label, shuffle=False):
            label_list_, seq_len_list_ = self.predict_one_batch(sess, seqs)
            label_list.extend(label_list_)
            seq_len_list.extend(seq_len_list_)
        return label_list, seq_len_list

    def predict_one_batch(self, sess, seqs):
        """

        :param sess:
        :param seqs:
        :return: label_list
                 seq_len_list
        """
        feed_dict, seq_len_list = self.get_feed_dict(seqs, dropout_keep_prob=1.0)

        if self.CRF:
            logits, transition_params = sess.run([self.logits, self.transition_params],
                                                 feed_dict=feed_dict)
            label_list = []
            for logit, seq_len in zip(logits, seq_len_list):
                viterbi_seq, _ = viterbi_decode(logit[:seq_len], transition_params)
                label_list.append(viterbi_seq)
            return label_list, seq_len_list
        else:
            label_list = sess.run(self.labels_softmax_, feed_dict=feed_dict)
            return label_list, seq_len_list

    def demo_one(self, sess, sent):
        label_list = []
        for seqs, labels in self.dd.batch_yield(sent, self.batch_size, self.word2id, self.tag2label, shuffle=False):
            label_list_, _ = self.predict_one_batch(sess, seqs)
            label_list.extend(label_list_)
        label2tag = {}
        for tag, label in self.tag2label.items():
            label2tag[label] = tag if label != 0 else label
        tag = [label2tag[label] for label in label_list[0]]
        return tag


    # def evaluate(self, label_list, seq_len_list, data, epoch=None):
    #     label2tag = {}
    #     for tag, label in self.tag2label.items():
    #         label2tag[label] = tag if label != 0 else label
    #
    #     model_predict = []
    #     for label_, (sent, tag) in zip(label_list, data):
    #         tag_ = [label2tag[label__] for label__ in label_]
    #         sent_res = []
    #         if len(label_) != len(sent):
    #             print(sent)
    #             print(len(label_))
    #             print(tag)
    #         for i in range(len(sent)):
    #             sent_res.append([sent[i], tag[i], tag_[i]])
    #         model_predict.append(sent_res)
    #     # print(model_predict)
    #     epoch_num = str(epoch + 1) if epoch != None else 'test'
    #     # print(epoch_num)
    #     label_path = os.path.join(self.result_path, 'label_' + epoch_num)
    #     metric_path = os.path.join(self.result_path, 'result_metric_' + epoch_num)
    #     # for _ in conlleval(model_predict, label_path, metric_path):
    #     #     self.logger.info(_)



