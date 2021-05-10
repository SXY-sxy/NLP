# -*- coding:utf-8 -*-
import argparse
import time
import flask
import tensorflow as tf
from flask import request
from gevent import pywsgi
from tensorflow.contrib.crf import viterbi_decode

from fenci.predict import Predict

date = "2021-05-08"
parser = argparse.ArgumentParser("BiLSTM-CRF for chinese split word task")
parser.add_argument("--vocab_path", type=str, default="./data/word2id.pkl", help="vocab table")
parser.add_argument("--trans_model_path", type=str, default=f"./trans_model/{date}/", help="export model path")
parser.add_argument("--CRF", type=bool, default=True, help="use or not crf")
parser.add_argument("--dropout_keep_prob", type=float, default=0.8, help="dropout keep_prob")
args = parser.parse_args()

sess=tf.Session()
global model
global graph

predict = Predict(args)
word2id = predict.read_vocab()

model = tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], args.trans_model_path)
signature = model.signature_def

app = flask.Flask(__name__)
@app.route("/predicts", methods=["GET","POST"])
def predicts():
    if request.method == 'POST':
        text = request.form.get('text')
    else:
        text = request.args.get('text')

    graph = tf.get_default_graph()
    with graph.as_default():
        print("-------------------going to restore checkpoint-------------------")
        feed_dict, seq_len_list, sencente = predict.text_processing(text, word2id)
        logits, trans = predict.predict(sess, feed_dict, signature)

        result_list = []
        for logit, seq_len, sent in zip(logits, seq_len_list, sencente):
            viterbi_seq, _ = viterbi_decode(logit[:seq_len], trans)
            result_list.append(predict.cut(sent, viterbi_seq))

        return str(result_list)


if __name__ == "__main__":
    server = pywsgi.WSGIServer(('0.0.0.0', 12345), app)
    server.serve_forever()
    # app.run(host='0.0.0.0', port=12345, debug=True)
