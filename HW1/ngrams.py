from collections import Counter

UNK = '<UNK>'

class NGram():
    """ 
    Args:
        sentence_list (list): The list of sentences (strings) to TRAIN the ngram models on. Only use the training data.
    
    For example,
    ['the twins tied the score in the sixth inning when reno bertoia beat out a high
     chopper to third base and scored on lenny greens double to left . <STOP>',
     'the white sox had taken a  lead in the top of the sixth on a pair of pop fly h
    its  a triple by roy sievers and single by camilo carreon  a walk and a sacrific
    e fly . <STOP>', ...]
    """
    
    def __init__(self, sentence_list, unk_thresh=1, add_k=0.0000001):
        
        self.unigram_count = Counter()
        self.unigram_count_w_UNK = Counter()
        self.bigram_count_w_UNK = Counter()
        self.trigram_count_w_UNK = Counter()
        self.p_mle = {}
        self.unk_thresh = unk_thresh
        self.count123Grams(sentence_list)
        self.bigram_dev_not_seen_in_training = 0.0
        self.trigram_dev_not_seen_in_training = 0.0
        self.total_num_of_unigrams_in_training = sum(self.unigram_count_w_UNK.values())
        self.num_of_unique_unigram_in_training = len(self.unigram_count_w_UNK.keys())
        self.num_of_unique_bigram_in_training = len(self.bigram_count_w_UNK.keys())
        self.num_of_unique_trigram_in_training = len(self.trigram_count_w_UNK.keys())
        self.add_k = add_k

    def tokensToNgramList(self, token_list, N):
        return [tuple(token_list[i:i + N]) for i in range(0, len(token_list) - N + 1)]


    def count123Grams(self, sentence_list):
        #First count the unigram
        for sentence in sentence_list:
            tokens_list = sentence.split()
            unigram_tuples = self.tokensToNgramList(tokens_list, 1)
            self.unigram_count.update(unigram_tuples)
            
        for sentence in sentence_list:
            tokens_list = sentence.split()
            #Replace rare words into <UNK>
            unked_tokens_list = self.replace_words_with_unk(tokens_list)
            #Create tuples for bigram and trigram
            unigram_tuples = self.tokensToNgramList(unked_tokens_list, 1)
            bigram_tuples = self.tokensToNgramList(unked_tokens_list, 2)
            trigram_tuples = self.tokensToNgramList(unked_tokens_list, 3)
            #Count the n-grams after replacing <UNK>
            self.unigram_count_w_UNK.update(unigram_tuples)
            self.bigram_count_w_UNK.update(bigram_tuples)
            self.trigram_count_w_UNK.update(trigram_tuples)

        # Let's check the most frequent n-grams
        self.unigram_top30 = self.unigram_count_w_UNK.most_common(30)
        self.bigram_top30 = self.bigram_count_w_UNK.most_common(30)
        self.trigram_top30 = self.trigram_count_w_UNK.most_common(30)
        #print (self.unigram_top30)


    def replace_words_with_unk(self, tokens_list):
        #print(tokens_list)
        for i in range(0, len(tokens_list)):
            if self.unigram_count[(tokens_list[i],)] <= self.unk_thresh:
                tokens_list[i] = UNK
        #print(tokens_list)
        return tokens_list

        
    def count(self, token_tuple):
        if len(token_tuple) == 1:
            return 1.0 * self.unigram_count_w_UNK[token_tuple]
        elif len(token_tuple) == 2:
            return 1.0 * self.bigram_count_w_UNK[token_tuple]
        elif len(token_tuple) == 3:
            return 1.0 * self.trigram_count_w_UNK[token_tuple]


    def pMLE(self, token_tuple):
        if len(token_tuple) == 1:
            if token_tuple not in self.p_mle:
                self.p_mle[token_tuple] = self.count(token_tuple) / self.total_num_of_unigrams_in_training 
                #self.p_mle[token_tuple] = (self.count(token_tuple)+add_k) / (self.total_num_of_unigrams_in_training  + add_k*self.num_of_unique_unigram_in_training)
            return self.p_mle[token_tuple]
        
        elif len(token_tuple) == 2:
            if token_tuple not in self.p_mle:
                # All unigrams are seen throughout the oov process. We don't need to worry about the denominator in bigram.
                self.p_mle[token_tuple] = (self.count(token_tuple) + self.add_k) / (self.count((token_tuple[0],)) + self.add_k*self.num_of_unique_unigram_in_training) 
                #print(token_tuple,self.p_mle[token_tuple]) 
            return self.p_mle[token_tuple]
        
        elif len(token_tuple) == 3:
            if token_tuple not in self.p_mle:
                self.p_mle[token_tuple] = (self.count(token_tuple) + self.add_k) / (self.count(tuple(token_tuple[0:2])) + self.add_k*self.num_of_unique_unigram_in_training)
                #print(token_tuple,self.p_mle[token_tuple])
            return self.p_mle[token_tuple]
