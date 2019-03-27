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


def get_dist_from_lda(article, beta):
  """ Get the topic distribution for the given document. 
  Args:
    article: String, original document
    beta
  Returns:
    tokenized_article
    topic_dist_str
    word2topic_dist_str
  """
  tokenized_article = jieba.cut(article)
  tokenized_article_str = ' '.join(tokenized_article)
  processed_article = [w for w in tokenized_article_str.split() if w not in stopwords]
  doc2bow = dictionary.doc2bow(processed_article)
  doc_topic = ldamodel.get_document_topics(doc2bow, minimum_probability=0, minimum_phi_value=None, per_word_topics=True)
  # Topic distribution for the whole document.
  topic_distribution = doc_topic[0]
  # Most probable topics per word. Each element in the list is a pair of a word’s id, and a list of topics sorted by their relevance to this word. 
  word_topics = doc_topic[2]
  word_topics_pair = {} #Change to dictionary
  for item in word_topics:
    word_topics_pair[item[0]] = item[1]
  topic_word_ids = word_topics_pair.keys()

  # Get the most relative topic
  topic_distribution_sorted = sorted(topic_distribution, key = lambda k:k[1], reverse = True)
  most_relative_topic = []
  sum_prob = float(0)
  for topic in topic_distribution_sorted:
    most_relative_topic.append(topic)
    sum_prob += topic[1]
    if sum_prob > beta:
      break
  top_topic_ids = [t[0] for t in most_relative_topic]
  top_topic = [str(t[0])+'-'+str(t[1]) for t in most_relative_topic]

  # Get word_to_topic_relevance_value
  word2topic_str = []
  article_ids = dictionary.doc2idx(tokenized_article_str.split())
  for idx in article_ids:
    if idx in topic_word_ids:
      item = word_topics_pair[idx]
      topic_and_phi = [float(0)]*FLAGS.num_topic
      for t in item:
        topic_id = t[0]
        if topic_id in top_topic_ids:
          topic_and_phi[topic_id] = t[1]
      topic_and_phi = [str(val) for val in topic_and_phi]
      word2topic_str.append('/'.join(topic_and_phi))
    else:
      word2topic_str.append('[stopwords]')

  if len(word2topic_str) != len(article_ids):
    print('length error!')

  return tokenized_article_str, '/'.join(top_topic), ' '.join(word2topic_str)