# -*-coding:utf-8-*-
import os
import re
import pickle
import numpy as np
import pandas as pd


class Data:
    def __init__(self, args):
        self.corpus_path = args.corpus_path
        self.train_path = args.train_path
        self.test_path = args.test_path
        self.vocab_path = args.vocab_path
        self.embedding_dim = args.hidden_dim


    # 语料库预处理
    def tag_line(self, words):
        # 对应字数组
        chars = []
        # 字数组对应标签
        tags = []
        temp_word = ''  # 用于合并组合词([]中的组合词)
        # print("+==============+++++++++++++++++++++++++++++++++++++=")
        for word in words:
            # print(word)
            if word == "/w": continue  # 去除因分句剩下的标点的标注
            word = word.strip('\t ')  # 如：迈向/v
            if temp_word == '':
                bracket_pos = word.find('[')  # [ ]ns
                w = word.split('/')[0]  # w:词；h:词性
                if bracket_pos == -1:  # 没有'['
                    if len(w) == 0: continue
                    chars.extend(w)  # 加入数组
                    # if h == 'ns':  # 词性为地名
                    # 单字词：+S；非单字词：+B(实体首部)、M*(len(w)-2)(实体中部)、E(实体尾部)
                    tags += ['S'] if len(w) == 1 else ['B'] + ['M'] * (len(w) - 2) + ['E']
                else:  # 有'['
                    # 获取'['后的词
                    w = w[bracket_pos + 1:]
                    temp_word += w
            else:
                bracket_pos = word.find(']')
                w = word.split('/')[0]
                if bracket_pos == -1:
                    temp_word += w
                else:
                    w = temp_word + w
                    h = word[bracket_pos + 1:]
                    temp_word = ''
                    if len(w) == 0: continue
                    chars.extend(w)
                    # if h == 'ns':
                    tags += ['S'] if len(w) == 1 else ['B'] + ['M'] * (len(w) - 2) + ['E']

        assert temp_word == ''
        return (chars, tags)

    # 中文分句
    def cut_sentence(self, sentence):
        sentence_list = re.split("。|！|\!|？|\?", sentence)
        return sentence_list

    # 加载语料库
    def load_corpus(self):
        data = []
        train_data = []
        test_data = []
        pos = 0
        with open(self.corpus_path, encoding='utf-8') as corpus_f:
                # open(self.train_path, encoding="utf-8") as train_f, \
                # open(self.test_path, encoding="utf-8") as test_f:
            for line in corpus_f:
                line = line.strip('\r\n\t')
                sentence = self.cut_sentence(line)
                if sentence == '': continue
                for sent in sentence:
                    # 去除每行开始时间
                    if len(sent.split()) <= 1: continue  # 过滤空字符和仅有分句的标点符号
                    if sent.split()[0].split("/")[1] == 't':
                        words = sent.split()[1:]  # 获取每行第1个及后面元素(去除每行开始时间)
                    else:
                        words = sent.split()

                    # line_chars:名 ；line_tags:对应标签(B/M/S/O)
                    line_chars, line_tags = self.tag_line(words)
                    data.append((line_chars, line_tags))

                    # 抽样20%作为测试集使用
                    if pos % 5 == 0:
                        test_data.append((line_chars, line_tags))
                    else:
                        train_data.append((line_chars, line_tags))
                    # isTest = True if pos % 5 == 0 else False
                    # saveObj = test_f if isTest else train_f
                    # for k, v in enumerate(line_chars):
                    #     saveObj.write(v + '\t' + line_tags[k] + '\n')
                    # saveObj.write('\n')
                    pos += 1
        return data, train_data, test_data

    # 建立词汇表
    def vocab_build(self):
        data, _, _ = self.load_corpus()
        word2id = {}

        # 统计词频, 并设置索引
        for words, tags in data:
            for word in words:
                if word not in word2id:
                    word2id[word] = [len(word2id)+1, 1]
                else:
                    word2id[word][1] += 1

        # 筛选出低频词，并删除
        low_freq_words = []
        for word, [word_index, word_freq] in word2id.items():
            if word_freq < 1:
                low_freq_words.append(word)
        for word in low_freq_words:
            del word2id[word]

        new_index = 1
        for word in word2id.keys():
            word2id[word] = new_index
            new_index += 1
        with open(self.vocab_path, 'wb') as fw:
            pickle.dump(word2id, fw)
        return word2id

    # 读取词汇表
    def read_vocab(self):
        with open(self.vocab_path, "rb") as fn:
            word2id = pickle.load(fn)
        print("vocab size:", len(word2id))
        return word2id


    # 生成词向量
    def random_embedding(self, word2id):
        """
        随机生成词嵌入，范围[-0.25, 0.25], shape = [词汇数量，embedding_dim]
        vocab : 词汇表
        embedding_dim : 词嵌入纬度
        """
        embedding_mat = np.random.uniform(-0.25, 0.25, (len(word2id), self.embedding_dim))
        embedding_mat = np.float32(embedding_mat)
        return embedding_mat
    ## 预训练词向量


    # 句子在词汇表中的id
    def sentence2id(self, sent, word2id):
        sentence_id = []
        for word in sent:
            if word not in word2id:
                word = '<UNK>'
            sentence_id.append(word2id[word])
        return sentence_id

    def pad_sequences(self, sequences, pad_mark=0):
        # 句子的最大长度
        max_len = max(map(lambda x: len(x), sequences))
        seq_list, seq_len_list = [], []
        for seq in sequences:
            seq = list(seq)
            seq_ = seq[:max_len] + [pad_mark] * max(max_len - len(seq), 0)
            seq_list.append(seq_)
            seq_len_list.append(min(len(seq), max_len))
        return seq_list, seq_len_list


    # 将句子转化为对应的词id，标注转化为对应的数字
    def batch_yield(self, data, batch_size, word2id, tag2label, shuffle=False):
        if shuffle:
            np.random.shuffle(data)

        seqs, labels = [], []
        for (sent_, tag_) in data:
            sent_id = self.sentence2id(sent_, word2id)
            label_id = [tag2label[tag] for tag in tag_]

            if len(seqs) == batch_size:
                yield seqs, labels
                seqs, labels = [], []

            seqs.append(sent_id)
            labels.append(label_id)

        if len(seqs) != 0:
            yield seqs, labels




if __name__=="__main__":
    dd = Data("data/2014_corpus.txt")
    data, _, _ = dd.load_corpus()
    # print(data)
    # dd.vocab_build()  # 建立词汇表
    with open("word2id.pkl", 'rb') as fr:
        word2id = pickle.load(fr)
    print(word2id)

    # 生成随机向量
    embedding_mat = dd.random_embedding(word2id)
    # 根据词汇表生成句子对应的word_id
    sentence_id = dd.sentence2id("我爱北京天安门", word2id)
    print(sentence_id)



