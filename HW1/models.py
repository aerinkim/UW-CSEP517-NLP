from preprocess import preprocess 
import math


class NGramLanguageModel():
    def __init__(self, nGram, add_k=0.0000001):
        #nGrams are always from the training set.
        self.nGram = nGram
        self.add_k = add_k

    def logPredictionSentence(self, token_list):
        # For example of unigram, 
        # log P(sentence) = log (P(w1)*P(w2)*...*P(wn)) = log P(w1) + log P(w2) + ... + log P(w3) 
        unigrams = self.nGram.tokensToNgramList(token_list, 1)
        bigrams = self.nGram.tokensToNgramList(token_list, 2)    
        trigrams = self.nGram.tokensToNgramList(token_list, 3)
        logUniProb = 0.0
        logBiProb = 0.0 
        logTriProb = 0.0

        for unigram in unigrams:
            logUniProb += math.log(self.nGram.pMLE(unigram, self.add_k), 2)
            #print (unigram, self.nGram.pMLE(unigram, self.add_k))

        for bigram in bigrams:
            logBiProb += math.log(self.nGram.pMLE(bigram, self.add_k), 2)
            #print (bigram, self.nGram.pMLE(bigram, self.add_k))

        for trigram in trigrams:
            logTriProb += math.log(self.nGram.pMLE(trigram, self.add_k), 2)
            #print (trigram, self.nGram.pMLE(trigram, self.add_k))

        return logUniProb, logBiProb, logTriProb

    def perplexity(self, test_file):
        # l = 1/M * \sigma log P(sentence)
        total_words_in_test_set = 0.0
        sum_of_prediction_uni = 0.0
        sum_of_prediction_bi = 0.0
        sum_of_prediction_tri = 0.0
        
        sentences = preprocess(test_file)
        # We are summing EVERY probability of the sentences
        for sentence in sentences:
            #print (sentence)
            unked_token_list = self.nGram.replace_words_with_unk(sentence.split())
            total_words_in_test_set += len(unked_token_list)

            log_prediction_uni, log_prediction_bi, log_prediction_tri = self.logPredictionSentence(unked_token_list)
            #print(log_prediction_uni, log_prediction_bi, log_prediction_tri)
            sum_of_prediction_uni += log_prediction_uni
            sum_of_prediction_bi += log_prediction_bi
            sum_of_prediction_tri += log_prediction_tri

        return math.pow(2, -1.0 / total_words_in_test_set * sum_of_prediction_uni), \
                math.pow(2, -1.0 / total_words_in_test_set * sum_of_prediction_bi), \
                math.pow(2, -1.0 / total_words_in_test_set * sum_of_prediction_tri)



class LinearInterpolation():
    def __init__(self, nGram, a1, a2, a3, add_k=0.0000001):
        self.nGram = nGram
        self.a1 = a1
        self.a2 = a2
        self.a3 = a3
        self.add_k = add_k

    def logPredictionSentence(self, token_list): 
        trigrams =  self.nGram.tokensToNgramList(token_list, 3)
        logProb = 0.0
        for trigram in trigrams:
            uniProb = self.a1 * self.nGram.pMLE((trigram[0],), self.add_k)
            biProb = self.a2 * self.nGram.pMLE(tuple(trigram[0:2]), self.add_k)
            triProb = self.a3 * self.nGram.pMLE(trigram, self.add_k)
            logProb += math.log(uniProb + biProb + triProb, 2)
        return logProb

    def perplexity(self, test_file):
        # l = 1/M * \sigma log P(sentence)
        total_words_in_test_set = 0.0
        sum_of_prediction_tri = 0.0
        
        sentences = preprocess(test_file)
        # We are summing EVERY probability of the sentences
        for sentence in sentences:
            #print (sentence)
            unked_token_list = self.nGram.replace_words_with_unk(sentence.split())
            total_words_in_test_set += len(unked_token_list)

            log_prediction_tri = self.logPredictionSentence(unked_token_list)

            sum_of_prediction_tri += log_prediction_tri

        return math.pow(2, -1.0 / total_words_in_test_set * sum_of_prediction_tri)

