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
from utils import *


def viterbi(vocab, vocab_tag, words, tags, t_bigram_count, t_unigram_count, e_bigram_count, e_unigram_count, ADD_K):
	vocab_size = len(vocab)
	V = [{}]

	for t in vocab_tag:
		# Prob of very first word
		prob = np.log2(float(e_bigram_count.get((words[0],t),0)+ADD_K))-np.log2(float(e_unigram_count[t]+vocab_size*ADD_K))
		# trigram V[0][0]
		V[0][t] = {"prob": prob, "prev": None}
	
	for i in range(1,len(words)):
		V.append({})
		for t in vocab_tag:
			V[i][t] =  {"prob": np.log2(0), "prev": None}
		for t in vocab_tag:
			max_trans_prob = np.log2(0);
			for prev_tag in vocab_tag:
				trans_prob = np.log2(float(t_bigram_count.get((t, prev_tag),0)+ADD_K))-np.log2(float(t_unigram_count[prev_tag]+vocab_size*ADD_K))	
				if V[i-1][prev_tag]["prob"]+trans_prob > max_trans_prob:
					max_trans_prob = V[i-1][prev_tag]["prob"]+trans_prob 
					max_prob = max_trans_prob+np.log2(e_bigram_count.get((words[i],t),0)+ADD_K)-np.log2(float(e_unigram_count[t]+vocab_size*ADD_K))
					V[i][t] = {"prob": max_prob, "prev": prev_tag}
	opt = []
	previous = None	
	max_prob = max(value["prob"] for value in V[-1].values())
	# Get most probable state and its backtrack
	for st, data in V[-1].items():
		if data["prob"] == max_prob:
			opt.append(st)
			previous = st
			break
	for t in range(len(V) - 2, -1, -1):
		opt.insert(0, V[t + 1][previous]["prev"])
		previous = V[t][previous]["prev"]
	return opt


def compute_accu(predict, ground):
	num = len(ground)
	correct = 0
	total = 0
	for i in range(0,num):
		total = total + 1
		if predict[i]==ground[i]:
			correct = correct + 1

	return (correct,total)


def bigram_inference(vocab, vocab_tag, dataset, t_bigram_count, t_unigram_count, e_bigram_count, e_unigram_count, ADD_K):
	conf_matrix = defaultdict(set)
	beginning_decoding = time.time();
	total = 0
	hit = 0
	line_counter = 0
	for line in dataset:
		line_counter = line_counter + 1
		words = [w[0] for w in line]
		tags = [t[1] for t in line]
		predict = viterbi(vocab, vocab_tag, words, tags, t_bigram_count, t_unigram_count, e_bigram_count, e_unigram_count, ADD_K)
		num = len(tags)
		for i in range(0,num):		
			conf_matrix[tags[i], predict[i]] = conf_matrix.get((tags[i], predict[i]),0) + 1

		(correct,count) = compute_accu(predict,tags)
		hit = hit + correct
		total = total + count
		#print ('computing.. (%d/%d): current accuracy = %d/%d = %f' % (line_counter,len(dataset),hit,total,hit/(float)(total)))
	print ('(add-k k=%f) bigram hmm accuracy = %f (%d/%d)' % (ADD_K,hit/(float)(total),hit,total))
	ending_decoding = time.time();
	print ("The decoding time is : ", ending_decoding - beginning_decoding)
	print ("#####################################")
	print ("######### confusion matrix###########")
	print ("#####################################")
	#print_confusion_matrix(conf_matrix,vocab_tag)


def main():

	# Test it on Train set.
	train_corpus = read_dataset(TRAIN_SET_PATH)
	for oov_thres in [1,5,10]:
		(train_data, vocab, tags) = preprocessing(train_corpus, oov_thres) #final_corpus,words_unk_counts,tags_unk_counts
		test_data = preproc_test(train_corpus,vocab)
		K_set = [1, 0.01, 0.0001, 0.000001]
		t_trigram_count, t_bigram_count, t_unigram_count, e_trigram_count, e_bigram_count, e_unigram_count = learning(train_data)
		for ADD_K in K_set:
			print('oov thres: ', oov_thres)
			bigram_inference(vocab, tags, test_data, t_bigram_count, t_unigram_count, e_bigram_count, e_unigram_count, ADD_K)

"""
	# Test it on Dev set.
	train_corpus = read_dataset(TRAIN_SET_PATH)
	test_corpus = read_dataset(DEV_SET_PATH)
	for oov_thres in [1,5,10]:
		(train_data, vocab, tags) = preprocessing(train_corpus, oov_thres) #final_corpus,words_unk_counts,tags_unk_counts
		test_data = preproc_test(test_corpus,vocab)
		K_set = [1, 0.01, 0.0001, 0.000001]
		t_trigram_count, t_bigram_count, t_unigram_count, e_trigram_count, e_bigram_count, e_unigram_count = learning(train_data)
		for ADD_K in K_set:
			print('oov thres: ', oov_thres)
			bigram_inference(vocab, tags, test_data, t_bigram_count, t_unigram_count, e_bigram_count, e_unigram_count, ADD_K)

"""

	# Train it on Train+Bonus set, Test it on Dev set.
	train_corpus = read_dataset(TRAIN_SET_PATH)
	bonus_corpus = read_dataset(BONUS_SET_PATH)
	train_corpus = train_corpus+bonus_corpus
	test_corpus = read_dataset(DEV_SET_PATH)
	for oov_thres in [1,5,10]:
		(train_data, vocab, tags) = preprocessing(train_corpus, oov_thres) #final_corpus,words_unk_counts,tags_unk_counts
		test_data = preproc_test(test_corpus,vocab)
		K_set = [1, 0.01, 0.0001, 0.000001]
		t_trigram_count, t_bigram_count, t_unigram_count, e_trigram_count, e_bigram_count, e_unigram_count = learning(train_data)
		for ADD_K in K_set:
			print('oov thres: ', oov_thres)
		

if __name__ == "__main__":
	main() 