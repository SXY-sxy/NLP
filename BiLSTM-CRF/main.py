#-*-coding:utf-8-*-
import os
import tensorflow as tf
import numpy as np
import os, argparse, time, random
from fenci.embedding import Data
from fenci.bilstmcrf import BiLSTM_CRF
from fenci.model_flask_deploy import DeployModel

# 会话配置
from fenci.predict import Predict


def tensorflow_conf():
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # default: 0

    # 对tensorflow中sesson配置
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.4  # need ~700MB GPU memory
    return config


if __name__=="__main__":
    date = time.strftime("%Y-%m-%d", time.localtime())
    # date = "2021-04-10"
    tf_config = tensorflow_conf()
    parser = argparse.ArgumentParser("BiLSTM-CRF for chinese split word task")
    parser.add_argument("--mode", type=str, default="deploy", help="train/export/predict/deploy")
    # path
    parser.add_argument("--corpus_path", type=str, default="./data/2014_corpus.txt", help="cropus for 人民日报")
    parser.add_argument("--train_path", type=str, default="./data/train.txt", help="trainset path")
    parser.add_argument("--test_path", type=str, default="./data/test.txt", help="testset path")
    parser.add_argument("--vocab_path", type=str, default="./data/word2id.pkl", help="vocab table")
    parser.add_argument("--model_path", type=str, default=f"./checkpoints/{date}/", help="model path")
    parser.add_argument("--ckpt_prefix", type=str, default=f"./checkpoints/{date}/model.ckpt", help="checkpoints_prefix path")
    parser.add_argument("--trans_model_path", type=str, default=f"./trans_model/{date}/", help="export model path")
    parser.add_argument("--pro_vocab_path", type=str, default="./data/word2id.pkl", help="already exists vocab tabel")

    # parameter
    parser.add_argument("--batch_size", type=int, default=100, help="sample of each minibatch")
    parser.add_argument("--epoch", type=int, default=2, help="epoch of training")
    parser.add_argument("--hidden_dim", type=int, default=300, help="dim of hidden state")
    parser.add_argument("--optimizer", type=str, default="Adam", help="optimizer funtion:Adam/Adadelta/Adagrad/RMSProp/Momentum/SGD")
    parser.add_argument("--learning_rate", type=float, default=0.01, help="learning rate")
    parser.add_argument("--clip", type=float, default=5.0, help="gradient clipping")  #-------------
    parser.add_argument("--dropout_keep_prob", type=float, default=0.8, help="dropout keep_prob") #--------------

    # bool
    parser.add_argument("--CRF", type=bool, default=True, help="use or not crf")
    parser.add_argument("--shuffle", type=bool, default=True, help="shuffle or not data")
    parser.add_argument("--update_embedding", type=bool, default=True, help="update or not embeddings")
    parser.add_argument("--use_pro_emb", type=bool, default=False, help="use or not exists embeddings")
    args = parser.parse_args()

    dd = Data(args)

    # 检查路径是否存在，并建立不存在路径
    if not os.path.exists(args.model_path):
        os.mkdir(args.model_path)
    if not os.path.exists(args.trans_model_path):
        os.mkdir(args.trans_model_path)

    # 读取语料库
    corpus, train, test = dd.load_corpus()
    train = train[:1000]
    test = test[:200]

    # 创建/读取词汇表
    if os.path.exists(args.vocab_path):
        word2id = dd.read_vocab()
    else:
        word2id = dd.vocab_build()

    global embeddings
    if not args.use_pro_emb:
        # 随机生成词嵌入
        embeddings = dd.random_embedding(word2id)
        log_pre = "not use pretrained embeddings"
    # else:
    #     embeddings = embeddings

    # 对应的字符转为数字
    tag2label = {'s': 0, 'S': 0, 'b': 1, 'B': 1, 'm': 2, 'M': 2, 'e': 3, 'E': 3}

    # 运行模式
    if args.mode == "train":
        model = BiLSTM_CRF(args, embeddings, tag2label, word2id, tf_config)
        model.build_graph()
        print("train data: {}".format(len(train)))
        print("test data: {}".format(len(test)))
        model.train(train=train, dev=test)
    elif args.mode == "predict":
        test = Predict(args=args)
        text = "我爱北京天安门。我喜欢打游戏！"
        result_list = test.model_split(text)
        print(result_list)
    elif args.mode == "export":
        model = BiLSTM_CRF(args, embeddings, tag2label, word2id, tf_config)
        model.build_graph()
        saver = tf.train.Saver()
        with tf.Session(config=tf_config) as sess:
            saver.restore(sess, tf.train.latest_checkpoint(args.model_path))

            # 创建一个builder，并将导出模型路径加载进来
            builder = tf.saved_model.builder.SavedModelBuilder(args.trans_model_path)

            # 将输入及输出张量与名称挂钩
            inputs = {
                'word_ids': tf.saved_model.utils.build_tensor_info(model.word_ids),
                'sequence_lengths': tf.saved_model.utils.build_tensor_info(model.sequence_lengths),
                'dropout_pl': tf.saved_model.utils.build_tensor_info(model.dropout_pl)
            }

            outputs = {
                'logits': tf.saved_model.utils.build_tensor_info(model.logits),
                'transition_params': tf.saved_model.utils.build_tensor_info(model.transition_params)
            }

            # 签名定义
            class_signautre_def = tf.saved_model.signature_def_utils.build_signature_def(
                inputs=inputs,
                outputs=outputs,
                method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
            )

            # 加入运行图中
            builder.add_meta_graph_and_variables(
                sess,
                [tf.saved_model.tag_constants.SERVING],
                signature_def_map={
                    'class_def': class_signautre_def
                }
            )
            builder.save()
    else:
        pass


