from ngrams import *
from models import *
from preprocess import *

train_tokens_list = preprocess('data/brown.train.txt') 
train_file = 'data/brown.train.txt'
dev_file = 'data/brown.dev.txt'
test_file = 'data/brown.test.txt' 


####################
# 3. Language Models 
####################

# Chosing the right oov threshold using the dev_set

print("\n ####### Reproducing problem 3. table 1 #######") 
for unk_thres in [1,5,10,20,30]:
	print("UNK threshold: ", unk_thres)
	#ngram should be declared with train set.
	ngram = NGram(train_tokens_list, unk_thres)
	model = NGramLanguageModel(ngram)
	unigram_score, bigram_score, trigram_score = model.perplexity(dev_file)
	print ("unigram_score, bigram_score, trigram_score: ", unigram_score, bigram_score, trigram_score)

print("\n ####### To reproduce problem 3. table 2, just remove lower() in preprocess.py. #######")

print("\n ####### Reproducing problem 3. table 3 #######") 
unk_thres = 1
ngram = NGram(train_tokens_list, unk_thres)
for files_to_test in ['data/brown.train.txt','data/brown.dev.txt','data/brown.test.txt']:
	model = NGramLanguageModel(ngram)
	unigram_score, bigram_score, trigram_score = model.perplexity(files_to_test)
	print ('perplexity score for ',files_to_test)
	print ("unigram_score, bigram_score, trigram_score: ", unigram_score, bigram_score, trigram_score)


##############
# 4. Smoothing
##############


print("\n ####### 4-1-a. Report perplexity scores on training and dev sets for various values of K (no interpolation). #######") 

unk_thres = 1

for files_to_test in ['data/brown.train.txt','data/brown.dev.txt']:
	for k in [10, 1, 0.1, 0.001, 0.00001]:
		ngram = NGram(train_tokens_list, unk_thres, k)
		model = NGramLanguageModel(ngram)
		unigram_score, bigram_score, trigram_score = model.perplexity(files_to_test)
		print ('perplexity score for ', files_to_test, " using add-K as ", ngram.add_k)
		print ("unigram_score, bigram_score, trigram_score: ", unigram_score, bigram_score, trigram_score)


print("\n ####### 4-1-b. report perplexity scores on training and dev sets for various values of l1, l2, l3 (no K smoothing).#######") 

unk_thres = 1
ngram = NGram(train_tokens_list, unk_thres)
for files_to_test in ['data/brown.train.txt','data/brown.dev.txt']:
	for lambda_set in [(0.3333,0.3333,0.3333),(0.1,0.1,0.8),(0.1,0.8,0.1),(0.8,0.1,0.1),(0.2,0.3,0.5)]:
		model = LinearInterpolation(ngram,lambda_set[0],lambda_set[1],lambda_set[2])
		perplexity = model.perplexity(files_to_test)
		print ('perplexity score for ', files_to_test ," with the lambda_set of ", lambda_set)
		print ("perplexity score: ", perplexity)



print("\n ####### 4-1-c. Putting it all together, report perplexity on the test set, using different smoothing techniques and the corresponding hyper-parameters that you chose from the dev set.#######") 

best_lambda_set = (0.3333,0.3333,0.3333)
best_k =  0.001

unk_thres = 1
ngram = NGram(train_tokens_list, unk_thres, best_k)
model = LinearInterpolation(ngram, best_lambda_set[0], best_lambda_set[1], best_lambda_set[2])
perplexity = model.perplexity(test_file)
print ('perplexity score for ', test_file ," with the lambda_set of ", best_lambda_set)
print ("perplexity score: ", perplexity)


print("\n ####### Appendix #######")
print("\n Top 30 unigrams in training data", ngram.unigram_top30)
print("\n Top 30 bigrams in training data", ngram.bigram_top30)
print("\n Top 30 trigrams in training data", ngram.trigram_top30)