#coding=utf-8
import os
import time
import itertools
import sys
import numpy as np
import tensorflow as tf
import udc_model
import udc_hparams
import udc_metrics
import udc_inputs
from models.dual_encoder import dual_encoder_model
from models.helpers import load_vocab

tf.flags.DEFINE_string("model_dir", "./jinfu_model/", "Directory to load model checkpoints from")
tf.flags.DEFINE_string("vocab_processor_file", "./jinfu_data/vocab_processor.bin", "Saved vocabulary processor file")
FLAGS = tf.flags.FLAGS

if not FLAGS.model_dir:
  print("You must specify a model directory")
  sys.exit(1)

def tokenizer_fn(iterator):
  return (x.split(" ") for x in iterator)

# Load vocabulary
vp = tf.contrib.learn.preprocessing.VocabularyProcessor.restore(
  FLAGS.vocab_processor_file)

# Load your own data here
#INPUT_CONTEXT = "兑 换 的 优 惠 券 不 能 用 怎 么 办"
INPUT_CONTEXT = "优 惠 券 不 能 用 怎 么 办 啊"
POTENTIAL_RESPONSES = ["您 好 ， 如 果 您 在 积 分 商 城 兑 换 的 优 惠 券 无 法 使 用 ， 您 可 以 打 开 小 米 金 融 A P P ， 右 上 角 礼 盒 图 标   →   「 积 分 商 城 」   →     「 兑 换 记 录 」 →     「 客 服 」 进 行 咨 询 处 理 。", "犹 豫 期 之 后 退 保 仅 退 还 保 单 的 现 金 价 值 ， 有 可 能 造 成 您 的 损 失 ， 请 您 谨 慎 操 作 。", "呵 呵 哈 哈 哈 或 或 或 或 或 或 或 或 或 或 或 或 或 或 或 或 或 或"]

def get_features(context, utterance):
  context_matrix = np.array(list(vp.transform([context])))
  utterance_matrix = np.array(list(vp.transform([utterance])))
  context_len = len(context.split(" "))
  utterance_len = len(utterance.split(" "))
  features = {
    "context": tf.convert_to_tensor(context_matrix, dtype=tf.int64),
    "context_len": tf.constant(context_len, shape=[1,1], dtype=tf.int64),
    "utterance": tf.convert_to_tensor(utterance_matrix, dtype=tf.int64),
    "utterance_len": tf.constant(utterance_len, shape=[1,1], dtype=tf.int64),
  }
  return features, None

if __name__ == "__main__":
  hparams = udc_hparams.create_hparams()
  model_fn = udc_model.create_model_fn(hparams, model_impl=dual_encoder_model)
  estimator = tf.contrib.learn.Estimator(model_fn=model_fn, model_dir=FLAGS.model_dir)

  # Ugly hack, seems to be a bug in Tensorflow
  # estimator.predict doesn't work without this line
  #estimator._targets_info = tf.contrib.learn.estimators.tensor_signature.TensorSignature(tf.constant(0, shape=[1,1]))

  print("Context: {}".format(INPUT_CONTEXT))
  sess = tf.Session()
  for r in POTENTIAL_RESPONSES:
    prob = estimator.predict(input_fn=lambda: get_features(INPUT_CONTEXT, r))
    #print("{}: {:g}".format(r, prob[0,0]))
    print("{}: {:g}".format(r, next(prob)[0]))
