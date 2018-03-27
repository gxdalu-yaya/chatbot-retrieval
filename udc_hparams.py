import tensorflow as tf
from collections import namedtuple

# Model Parameters
tf.flags.DEFINE_integer(
  "vocab_size",
  1933,
  "The size of the vocabulary. Only change this if you changed the preprocessing")

# Model Parameters
tf.flags.DEFINE_integer("embedding_dim", 64, "Dimensionality of the embeddings")
tf.flags.DEFINE_integer("rnn_dim", 64, "Dimensionality of the RNN cell")
tf.flags.DEFINE_integer("max_context_len", 30, "Truncate contexts to this length")
tf.flags.DEFINE_integer("max_utterance_len", 120, "Truncate utterance to this length")

# Pre-trained embeddings
#tf.flags.DEFINE_string("w2v_path", "./jinfu_data/vec.txt", "Path to pre-trained Glove vectors")
#tf.flags.DEFINE_string("vocab_path", "./jinfu_data/vocabulary.txt", "Path to vocabulary.txt file")
tf.flags.DEFINE_string("w2v_path", None, "Path to pre-trained Glove vectors")
tf.flags.DEFINE_string("vocab_path", None, "Path to vocabulary.txt file")

# Training Parameters
tf.flags.DEFINE_float("learning_rate", 0.0001, "Learning rate")
tf.flags.DEFINE_integer("batch_size", 64, "Batch size during training")
tf.flags.DEFINE_integer("eval_batch_size", 2, "Batch size during evaluation")
tf.flags.DEFINE_string("optimizer", "Adam", "Optimizer Name (Adam, Adagrad, etc)")

FLAGS = tf.flags.FLAGS

HParams = namedtuple(
  "HParams",
  [
    "batch_size",
    "embedding_dim",
    "eval_batch_size",
    "learning_rate",
    "max_context_len",
    "max_utterance_len",
    "optimizer",
    "rnn_dim",
    "vocab_size",
    "w2v_path",
    "vocab_path"
  ])

def create_hparams():
  return HParams(
    batch_size=FLAGS.batch_size,
    eval_batch_size=FLAGS.eval_batch_size,
    vocab_size=FLAGS.vocab_size,
    optimizer=FLAGS.optimizer,
    learning_rate=FLAGS.learning_rate,
    embedding_dim=FLAGS.embedding_dim,
    max_context_len=FLAGS.max_context_len,
    max_utterance_len=FLAGS.max_utterance_len,
    w2v_path=FLAGS.w2v_path,
    vocab_path=FLAGS.vocab_path,
    rnn_dim=FLAGS.rnn_dim)
