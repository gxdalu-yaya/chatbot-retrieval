#coding=utf-8
import array
import numpy as np
import tensorflow as tf
from collections import defaultdict
import io
import sys 
reload(sys)
sys.setdefaultencoding("utf-8")

def load_vocab(filename):
  vocab = None
  with io.open(filename, "r", encoding="utf-8") as f:
  #with open(filename, "r") as f:
    #vocab = [word.decode("utf-8") for word in f]
    vocab = f.read().splitlines()
  dct = defaultdict(int)
  for idx, word in enumerate(vocab):
    dct[word] = idx
  return [vocab, dct]

def load_w2v_vectors(filename, vocab):
  """
  Load w2v vectors from a .txt file.
  Optionally limit the vocabulary to save memory. `vocab` should be a set.
  """
  dct = {}
  vectors = array.array('d')
  current_idx = 0
  with io.open(filename, "r", encoding="utf-8") as f:
  #with open(filename) as f:
    for _, line in enumerate(f):
      tokens = line.strip().split(" ")
      word = tokens[0]
      entries = tokens[1:]
      if not vocab or word in vocab:
        dct[word] = current_idx
        '''
	entries_vec = list()
	for x in entries:
	  try:
	    entries_vec.append(float(x))
	  except ValueError,e:
	    print "error:", e, "on line", line, "word:", word, "entries", x 
	vectors.extend(entries_vec)
        '''
	vectors.extend(float(x) for x in entries)
        current_idx += 1
    word_dim = len(entries)
    num_vectors = len(dct)
    tf.logging.info("Found {} out of {} vectors in Glove".format(num_vectors, len(vocab)))
    return [np.array(vectors).reshape(num_vectors, word_dim), dct]


def build_initial_embedding_matrix(vocab_dict, w2v_dict, w2v_vectors, embedding_dim):
  initial_embeddings = np.random.uniform(-0.25, 0.25, (len(vocab_dict), embedding_dim)).astype("float32")
  for word, w2v_word_idx in w2v_dict.items():
    word_idx = vocab_dict.get(word)
    initial_embeddings[word_idx, :] = w2v_vectors[w2v_word_idx]
  return initial_embeddings
