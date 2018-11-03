import re
import json
from collections import defaultdict
import os
import time
import math
import pandas as pd
from math import log
import numpy as np
from sklearn.metrics import confusion_matrix

STOP_SYMBOL = 'STOP'
START_SYMBOL = 'START'
UNK_SYMBOL = 'UNK'
TRAIN_SET_PATH = 'data/twt.train.json'
DEV_SET_PATH = 'data/twt.dev.json'
TEST_SET_PATH = 'data/twt.test.json'    
BONUS_SET_PATH = 'data/twt.bonus.json' 

#oov_thres = 1 # replacing by all words that occur equal to or fewer than n times in the training set

def non_freq_words_UNK(dataset, wordfreq, oov_thres, replaceToken=UNK_SYMBOL):
    for line in dataset:
        for idx in line:
            if wordfreq[idx[0]] <= oov_thres:
                idx[0] = replaceToken
    return dataset

def test_set_UNKing(dataset, voc, replaceToken=UNK_SYMBOL):
    for line in dataset:
        for idx in line:
            if (voc[idx[0]] == 0):
                idx[0] = replaceToken
    return dataset

def find_element_in_list(element, list_element):
    try:
        index_element = list_element.index(element)
        return True
    except ValueError:
        return False

def read_dataset(filename):
    with open(filename,'r') as lines:
        dataset = [json.loads(line) for line in lines]
    lines.close()
    return dataset

def preprocessing(dataset, oov_thres):
    for line in dataset:    # add STOP/START 
        line.append([STOP_SYMBOL,STOP_SYMBOL])
    # count the number of words
    words_counts = defaultdict(lambda: 0)
    tags_counts =  defaultdict(lambda: 0)
    total_words = 0
    for line in dataset:
        for idx in line:
            w = idx[0] #word
            tag = idx[1] #tag
            words_counts[w] = words_counts[w]+1
            tags_counts[tag] = tags_counts[tag]+1
            total_words = total_words + 1

    vocab_with_oov = [w for w in words_counts.items()]
    vocab_with_oov_size = len(vocab_with_oov)
    print ('Vocabulary size before removing oov:', vocab_with_oov_size)
    
    # replace non-freq words with UNK
    final_corpus = non_freq_words_UNK(dataset, words_counts, oov_thres, replaceToken=UNK_SYMBOL)
    
    words_unk_counts = defaultdict(lambda: 0)
    tags_unk_counts =  defaultdict(lambda: 0)
    total_unk_words = 0
    for line in final_corpus:
        for idx in line:
            w = idx[0] #word
            tag = idx[1] #tag
            words_unk_counts[w] = words_unk_counts[w]+1
            tags_unk_counts[tag] = tags_unk_counts[tag]+1
            total_unk_words = total_unk_words + 1
    print ('Vocabulary size after removing oov (with UNK)',  len(words_unk_counts))
    #print words_unk_counts
    return (final_corpus,words_unk_counts,tags_unk_counts)


def preproc_test(dataset,voc):
    for line in dataset:
        line.append([STOP_SYMBOL,STOP_SYMBOL])
    final_corpus = test_set_UNKing(dataset, voc, UNK_SYMBOL)
    return final_corpus


def learning(dataset):
    trans_trigram_count = defaultdict(set)
    trans_bigram_count = defaultdict(set)
    trans_unigram_count = defaultdict(lambda: 0)
    emiss_trigram_count = defaultdict(set)
    emiss_bigram_count = defaultdict(set)
    emiss_unigram_count = defaultdict(lambda: 0)
    beginning_training = time.time();
    
    # count transitions and emissions. idx2, idx1, idx0 -> is the sequence of the words. 
    for tweet in dataset:
        idx1 = [START_SYMBOL,START_SYMBOL]
        idx2 = [START_SYMBOL,START_SYMBOL]
        for word in tweet:
            idx0 = word # word = [u'doubt', u'N']
            if idx0 != START_SYMBOL:
                w0 = idx0[0]
                tag0 = idx0[1]

                w1 = idx1[0]
                tag1 = idx1[1]
                
                w2 = idx2[0]
                tag2 = idx2[1]
                
                trans_unigram_count[tag0] += 1
                emiss_unigram_count[tag0] += 1
                
                trans_bigram_count[tag0, tag1] = trans_bigram_count.get((tag0, tag1), 0) + 1
                trans_trigram_count[tag0, tag1, tag2] = trans_trigram_count.get((tag0, tag1, tag2), 0) + 1
                
                emiss_bigram_count[w0, tag0] = emiss_bigram_count.get((w0, tag0), 0) + 1
                emiss_trigram_count[w0, tag0] = emiss_trigram_count.get((w0, tag0), 0) + 1 # same as bigram!
            idx2 = idx1
            idx1 = idx0 
    ending_training = time.time();
    print ("Training time is : ", ending_training - beginning_training)
    return (trans_trigram_count, trans_bigram_count, trans_unigram_count, emiss_trigram_count, emiss_bigram_count, emiss_unigram_count)


def display(V):
    # Print a table of steps from dictionary
    yield " ".join(("%12d" % i) for i in range(len(V)))
    for state in V[0]:
        yield "%.7s: " % state + " ".join("%.7s" % ("%f" % v[state]["prob"]) for v in V)


def print_confusion_matrix(conf_matrix,vocab_tag):
    dict1 = {u'!': 0,
             u'#': 1,
             u'$': 2,
             u'&': 3,
             u',': 4,
             u'@': 5,
             u'A': 6,
             u'D': 7,
             u'E': 8,
             u'G': 9,
             u'L': 10,
             u'N': 11,
             u'O': 12,
             u'P': 13,
             u'R': 14,
             u'S': 15,
             'STOP': 16,
             u'T': 17,
             u'U': 18,
             u'V': 19,
             u'X': 20,
             u'Y': 21,
             u'Z': 22,
             u'^': 23,
             u'~': 24}
    Matrix = [[0 for x in range(25)] for y in range(25)]          
    for tag1 in vocab_tag:
        for tag2 in vocab_tag:
            x = dict1[tag1]
            y = dict1[tag2]
            Matrix[x][y] = conf_matrix.get((tag1,tag2),0)
    dict1_keys=sorted([key for (key,value) in dict1.items()])
    print (pd.DataFrame(Matrix,dict1_keys,dict1_keys))
