#coding=utf-8
import os
import time
import itertools
import sys
import io
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
def load_querylist(filename):
  INPUT_CONTEXT_LIST = [" ".join(line.strip().decode("utf-8")) for line in open(filename, "r")]
  return INPUT_CONTEXT_LIST

#INPUT_CONTEXT = "优 惠 券 不 能 用 怎 么 办 啊"

def get_features(context, utterances):
  context_matrix = np.array(list(vp.transform([context])))
  utterance_matrix = np.array(list(vp.transform([utterances[0]])))
  context_len = len(context.split(" "))
  utterance_len = len(utterances[0].split(" "))
  features = {
    "context": tf.convert_to_tensor(context_matrix, dtype=tf.int64),
    "context_len": tf.constant(context_len, shape=[1,1], dtype=tf.int64),
    "utterance": tf.convert_to_tensor(utterance_matrix, dtype=tf.int64),
    "utterance_len": tf.constant(utterance_len, shape=[1,1], dtype=tf.int64),
    "len": len(utterances)
  }
  for i in range(1, len(utterances)):
    utterance = utterances[i]
    utterance_matrix = np.array(list(vp.transform([utterance])))
    utterance_len = len(utterance.split(" "))
  
    features["utterance_{}".format(i)] = tf.convert_to_tensor(utterance_matrix, dtype=tf.int64)
    features["utterance_{}_len".format(i)] = tf.constant(utterance_len, shape=[1,1], dtype=tf.int64)
  return features, None

if __name__ == "__main__":
  hparams = udc_hparams.create_hparams()
  model_fn = udc_model.create_model_fn(hparams, model_impl=dual_encoder_model)
  estimator = tf.contrib.learn.Estimator(model_fn=model_fn, model_dir=FLAGS.model_dir)
  INPUT_CONTEXT_LIST = load_querylist("./test/jinfu.q.test")
  POTENTIAL_RESPONSES =  load_querylist("./test/jinfu.answer")
  POTENTIAL_RESPONSES = [response.encode("utf-8") for response in POTENTIAL_RESPONSES]
  #POTENTIAL_RESPONSES = ["若 系 统 识 别 不 通 过 ， 会 为 您 转 跳 到 人 工 识 别 ， 您 需 要 提 供 手 持 身 份 证 的 自 拍 照 片 和 身 份 证 正 面 照 片 ， 我 们 的 工 作 人 员 会 在 1 - 3 个 工 作 日 内 完 成 审 核 。 审 核 完 成 后 ， 将 会 为 您 推 送 相 关 短 信 告 知 您 是 否 通 过 审 核 。 若 审 核 未 通 过 ， 说 明 您 当 前 不 具 备 贷 款 申 请 条 件 。 系 统 的 判 定 依 据 会 定 期 更 新 ， 建 议 您 过 段 时 间 在 来 试 试 。", "干 脆 贷 额 度 由 系 统 进 行 统 一 计 算 和 评 估 ， 当 系 统 检 测 到 您 的 账 户 使 用 习 惯 良 好 、 信 用 优 秀 ， 将 为 您 提 高 贷 款 >额 度 。", "很 抱 歉 ， 若 您 的 贷 款 首 页 没 有 显 示 额 度 ， 是 由 于 您 的 综 合 评 估 不 通 过 。 干 脆 贷 根 据 用 户 情 况 综 合 评 估 ， 会 实 时 >调 整 评 估 结 果 ， 当 系 统 检 测 到 您 的 账 户 使 用 习 惯 良 好 ， 将 有 可 能 为 您 恢 复 干 脆 贷 额 度 ， 建 议 后 期 关 注 。", "哈 哈 哈"]
  # Ugly hack, seems to be a bug in Tensorflow
  # estimator.predict doesn't work without this line
  #estimator._targets_info = tf.contrib.learn.estimators.tensor_signature.TensorSignature(tf.constant(0, shape=[1,1]))

  sess = tf.Session()
  for INPUT_CONTEXT in INPUT_CONTEXT_LIST:
    INPUT_CONTEXT = INPUT_CONTEXT.encode("utf-8")
    #INPUT_CONTEXT = "贷 款 如 何 提 升 信 用 卡 额 度"
    print("Context: {}".format(INPUT_CONTEXT.replace(" ", "")))
    prob = estimator.predict(input_fn=lambda: get_features(INPUT_CONTEXT, POTENTIAL_RESPONSES))
    result = next(prob)
    answerId = result.argmax(axis=0)
    print("{}: {:g}".format(POTENTIAL_RESPONSES[answerId].replace(" ", ""), result[answerId]))
