import os
import numpy as np
import tensorflow as tf
import jieba
from gensim.models import LdaModel
from gensim.corpora import Dictionary

FLAGS = tf.app.flags.FLAGS

ldamodel = LdaModel.load('../text_preprocess/lda_result/lda_65')
dictionary = Dictionary.load('../text_preprocess/lda_result/final_dictionary.dictionary')

def read_text_file(text_file):
  lines = []
  with open(text_file, "r", encoding="utf-8") as f:
    for line in f:
      if line.strip() != '': lines.append(line.strip())
  return lines

stopwords = read_text_file('../Chinese_stopwords_1893.txt')

def softmax(x):
  """Compute the softmax of vector x."""
  exp_x = np.exp(x)
  softmax_x = exp_x / np.sum(exp_x)
  return softmax_x 

def lda_distribution(doc):
  """Calculates the lda distributions of text.

  Args:
    doc: String, the document to be processed

  Returns:
    lda_dis: Array, lda distribution of text
  """
  doc_lines = text_precessing(doc)
  doc_bow = dictionary.doc2bow(doc_lines.strip().split())
  doc_lda = ldamodel[doc_bow]

  lda_dis = np.zeros(FLAGS.num_topic)
  for v in doc_lda:
    lda_dis[v[0]] = v[1]
  return lda_dis

def text_precessing(text):
  """ Document preprocessing

  Args:
    text: String, the document to be processed

  Returns:
    String, processed document
  """
  # tokenized
  wordLst = jieba.cut(text)
  # Filter stop words
  filtered = [w for w in wordLst if w not in stopwords]

  return " ".join(filtered)

def cal_similarity(s1, s2):
  """Calculates the cos similarity between two vectors.

  Args:
    s1: Array, vector
    s2: Array, vector

  Returns:
    sim: float, cos similarity.
  """
  try:
    # s1 = softmax(_s1)
    # s2 = softmax(_s2)
    sim = np.dot(s1, s2) / (np.linalg.norm(s1) * np.linalg.norm(s2))
  except ValueError:
    sim=0
  return sim