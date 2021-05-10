# -*- coding:utf-8 -*-
import re
import pickle
import tensorflow as tf
from tensorflow.contrib.crf import viterbi_decode

class Predict:
    def __init__(self, args):
        # self.vocab_path = "./data/word2id.pkl"
        # self.export_dir = "./trans_model"
        # self.dropout_keep_prob = 0.8
        # self.CRF = True

        self.vocab_path = args.vocab_path
        self.export_dir = args.trans_model_path
        self.dropout_keep_prob = args.dropout_keep_prob
        self.CRF = args.CRF

    # 加载词汇表
    def read_vocab(self):
        with open(self.vocab_path, "rb") as fn:
            word2id = pickle.load(fn)
        print("vocab size:", len(word2id))
        return word2id

    # 词转词id
    def sentence2id(self, sent, word2id):
        if len(sent) == 0: return None

        sentence_id = []
        for word in sent.strip():
            if word not in word2id:
                word = "<UNK>"
            sentence_id.append(word2id[word])
        return sentence_id

    # 中文分句
    def cut_sentence(self, sentence):
        """
        将文章分为若干个句子
        """
        sent = re.findall('[\u4e00-\u9fa5a-zA-Z0-9]+',sentence,re.S)
        sent_re = "".join(sent)
        sentence_list = re.split("。|！|\!|？|\?", sent_re)
        sentence_list = [sent for sent in sentence_list if sent != '']
        return sentence_list

    # 文本预处理
    def text_processing(self, text, word2id):
        """
        将文本根据标点符号切分为句子
        句子转为句子id
        不同的句子填充0是长度相等
        """
        sentence = self.cut_sentence(text)
        sentenceid_list = [self.sentence2id(sent, word2id) for sent in sentence]
        max_len = max(map(lambda x: len(x), sentenceid_list))
        seq_id_list, seq_len_list = [], []
        for seq in sentenceid_list:
            seq_ = seq[:max_len] + [0] * max(max_len - len(seq), 0)
            seq_id_list.append(seq_)
            seq_len_list.append(min(len(seq), max_len))

        feed_dict = {"word_ids": seq_id_list,
                     "sequence_lengths": seq_len_list,
                     "dropout_keep_prob": self.dropout_keep_prob}

        return feed_dict, seq_len_list, sentence

    def predict(self, sess, feed_dict, signature):
        # 输入张量及参数在图中的name
        word_ids_tensor_name = signature['class_def'].inputs['word_ids'].name
        sequence_lengths_tensor_name = signature['class_def'].inputs['sequence_lengths'].name
        dropout_tensor_name = signature['class_def'].inputs['dropout_pl'].name

        logits_tensor_name = signature['class_def'].outputs['logits'].name
        transition_params_name = signature['class_def'].outputs['transition_params'].name

        # get 图中的张量的value
        word_ids = sess.graph.get_tensor_by_name(word_ids_tensor_name)
        sequence_lengths = sess.graph.get_tensor_by_name(sequence_lengths_tensor_name)
        dropout_keep_prob = sess.graph.get_tensor_by_name(dropout_tensor_name)

        logits = sess.graph.get_tensor_by_name(logits_tensor_name)
        transition_params = sess.graph.get_tensor_by_name(transition_params_name)


        logits, trans = sess.run([logits,transition_params],
                                 feed_dict={
                                     word_ids: feed_dict['word_ids'],
                                     sequence_lengths: feed_dict['sequence_lengths'],
                                     dropout_keep_prob: feed_dict['dropout_keep_prob']}
                                 )
        return logits, trans

    def model_split(self, text):
        word2id = self.read_vocab()
        feed_dict, seq_len_list, sencente = self.text_processing(text, word2id)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            meta_graph_def = tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], self.export_dir)
            signature = meta_graph_def.signature_def
            logits, trans = self.predict(sess, feed_dict=feed_dict, signature=signature)
            # label_list = []
            result_list = []
            for logit, seq_len, sent in zip(logits, seq_len_list, sencente):
                viterbi_seq, _ = viterbi_decode(logit[:seq_len], trans)
                # label_list.append(viterbi_seq)
                result_list.append(self.cut(sent, viterbi_seq))
        return result_list

    def cut(self, sent, tagstr):
        result = ''
        for (character, tag) in zip(sent, tagstr):
            result += character
            if tag == 0 or tag == 3:
                result += ' '
        return result



if __name__=="__main__":
    test = Predict(args=None)

    # word2id = test.read_vocab()
    text = "我爱北京天安门。我喜欢打游戏！"
    result_list= test.model_split(text)


