from math import log, exp, sqrt
from collections import defaultdict
import argparse

from numpy import mean

import nltk
from nltk import FreqDist
from nltk import bigrams
from nltk.tokenize import TreebankWordTokenizer

kLM_ORDER = 2
kUNK_CUTOFF = 3 #3
kNEG_INF = -1e6

kSTART = "<s>"
kEND = "</s>"

def lg(x):
    return log(x) / log(2.0)

class BigramLanguageModel:

    def __init__(self, unk_cutoff, jm_lambda=0.6, dirichlet_alpha=0.1,
                 katz_cutoff=5, kn_discount=0.1, kn_concentration=1.0,
                 tokenize_function=TreebankWordTokenizer().tokenize,
                 normalize_function=lambda x: x.lower()):
        self._unk_cutoff = unk_cutoff
        self._jm_lambda = jm_lambda
        self._dirichlet_alpha = dirichlet_alpha
        self._katz_cutoff = katz_cutoff
        self._kn_concentration = kn_concentration
        self._kn_discount = kn_discount
        self._vocab_final = False

        self._tokenizer = tokenize_function
        self._normalizer = normalize_function
        
        # Add your code here!

        #setting unknow words with 0
        def defaultdict_handle():
            return 0

        def bigram_dict_handle():
            return defaultdict(defaultdict_handle)

        self._vocab_dict = defaultdict(defaultdict_handle)
        self._unigram_dict = defaultdict(defaultdict_handle)
        self._bigram_dict = defaultdict(defaultdict_handle)

        self._bigram_word_key = defaultdict(bigram_dict_handle)
        self._bigram_context_key = defaultdict(bigram_dict_handle)

        self._unk_token = 'UNK'
        self._unigram_count =0
        self._bigram_types_count = 0

    def train_seen(self, word, count=1):

        """
        Tells the language model that a word has been seen @count times.  This
        will be used to build the final vocabulary.
        """
        assert not self._vocab_final, \
            "Trying to add new words to finalized vocab"


        # Add your code here!            
        self._vocab_dict[self._normalizer(word)] += count


    def get_vocab_len(self):
        return len(self._unigram_dict)
        

    def tokenize(self, sent):
        """
        Returns a generator over tokens in the sentence.  

        You don't need to modify this code.
        """
        for ii in self._tokenizer(sent):
            yield ii
        
    def vocab_lookup(self, word):
        """
        Given a word, provides a vocabulary representation.  Words under the
        cutoff threshold shold have the same value.  All words with counts
        greater than or equal to the cutoff should be unique and consistent.
        """
        assert self._vocab_final, \
            "Vocab must be finalized before looking up words"

        # Add your code here
        # remove the word below threshold and adding the unknow token
        if (self._vocab_dict[word] < self._unk_cutoff ) and (word != kSTART) and (word != kEND):
            return self._unk_token

        return word

    def finalize(self):
        """
        Fixes the vocabulary as static, prevents keeping additional vocab from
        being added
        """

        # You probably do not need to modify this code
        self._vocab_final = True

    def tokenize_and_censor(self, sentence):
        """
        Given a sentence, yields a sentence suitable for training or
        testing.  Prefix the sentence with <s>, replace words not in
        the vocabulary with <UNK>, and end the sentence with </s>.

        You should not modify this code.
        """
        yield self.vocab_lookup(kSTART)
        for ii in self._tokenizer(sentence):
            yield self.vocab_lookup(self._normalizer(ii))
        yield self.vocab_lookup(kEND)


    def normalize(self, word):
        """
        Normalize a word

        You should not modify this code.
        """
        return self._normalizer(word)



    def mle(self, context, word):
        """
        MLE (Maximum Liklehod)
        Return the log MLE estimate of a word given a context.  If the
        MLE would be negative infinity, use kNEG_INF
        (((( WITH OUT THE USE of UNKOWN TOKEN))))
        """

        # This initially return 0.0, ignoring the word and context.
        # Modify this code to return the correct value.
        numerator = self._bigram_dict[context, word]
        denomirator = self._unigram_dict[context]


        if (numerator == 0) or (denomirator == 0):
            return kNEG_INF

        return lg(float(numerator)/denomirator)

    def laplace(self, context, word):
        """
        Return the log MLE estimate of a word given a context.
        """

        # This initially return 0.0, ignoring the word and context.
        # Modify this code to return the correct value.
        numerator = self._bigram_dict[context, word] + 1
        denomirator = self._unigram_dict[context] +  \
        len(self._unigram_dict)

        return lg(float(numerator)/denomirator)


    def jelinek_mercer(self, context, word):
        """
        Return the Jelinek-Mercer log probability estimate of a word
        given a context; interpolates context probability with the
        overall corpus probability.
        sum(all lambda) =1
        """
        # This initially return 0.0, ignoring the word and context.
        # Modify this code to return the correct value.
        
        # uingram probability
        unigram_prob = float(self._unigram_dict[word])/len(self._unigram_dict)
        # unigram_prob = float(self._unigram_dict[word])/self._unigram_count


        # bigram Probability
        bigram_prob = float(self._bigram_dict[context, word]) / \
                self._unigram_dict[context]


        return lg((1- self._jm_lambda) * unigram_prob +
                self._jm_lambda * bigram_prob)

    def kneser_ney(self, context, word):
        """
        Return the log probability of a word given a context given
        Kneser Ney backoff
        """
        # This initially return 0.0, ignoring the word and context.
        # Modify this code to return the correct value.
        d = self._kn_discount
        theta = self._kn_concentration

        prob1 = max(self._bigram_dict[context, word] - d, 0) / (
                self._unigram_dict[context] + theta)

        lam1 = (theta + d* len(self._bigram_context_key[context])) / ( 
            theta + self._unigram_dict[context])

        prob2 = max(len(self._bigram_word_key[word]) -d,0) / (
                self._bigram_types_count + theta)

        lam2 = (theta + d*len(self._bigram_word_key)) / (
                self._bigram_types_count + theta)

        prob3 = 1 / len(self._unigram_dict)

        p = prob1 + lam1 * (prob2 + lam2 * prob3)
        return lg(p)




    def dirichlet(self, context, word):
        """
        Add K smoothing
        Additive smoothing, assuming independent Dirichlets with fixed
        hyperparameter.
        """
        # This initially return 0.0, ignoring the word and context.
        # Modify this code to return the correct value.
        numerator = self._bigram_dict[context, word] + \
                self._dirichlet_alpha
        denomirator = self._unigram_dict[context] +  \
            self._dirichlet_alpha * float(len(self._unigram_dict))

        return lg(float(numerator)/denomirator)


    def add_train(self, sentence):
        """
        Add the counts associated with a sentence.
        """

        # You'll need to complete this function, but here's a line of
        # code that will hopefully get you started.

        for context, word in bigrams(self.tokenize_and_censor(sentence)):
            self._bigram_dict[context, word] +=1

            if self._bigram_word_key[word][context] == 0:
                self._bigram_types_count +=1
            self._bigram_word_key[word][context] +=1
            self._bigram_context_key[context][word] +=1

        for word in self.tokenize_and_censor(sentence):
            self._unigram_dict[word] +=1
            self._unigram_count += 1



    def perplexity(self, sentence, method):
        """
        Compute the perplexity of a sentence given a estimation method

        You do not need to modify this code.
        """
        return sqrt(2.0 ** (-1.0 * mean([method(context, word) for context, word in \
                                    bigrams(self.tokenize_and_censor(sentence))])))

    def sample(self, method, samples=25):
        """
        Sample words from the language model.
        
        @arg samples The number of samples to return.
        """
        # Modify this code to get extra credit.  This should be
        # written as an iterator.  I.e. yield @samples times followed
        # by a final return, as in the sample code.
        pass

        # for ii in range(samples):
        #     yield ""
        # return

    def tokenize_and_get_last_word(self, sent):
        """
        tokenize the sentence and return the last token
        """
        if sent == '':
            return self.vocab_lookup(kSTART)

        return self.vocab_lookup(self._tokenizer(sent)[-1])



    def predict_word(self, sent, method):

        context = self.tokenize_and_get_last_word(sent)
        last_word = ""
        max_prob = kNEG_INF

        for word in self._unigram_dict:

            prob = method(context, word)
            if (prob > max_prob) and (word != self._unk_token):
                last_word = word
                max_prob = prob

        return last_word




# You do not need to modify the below code, but you may want to during
# your "exploration" of high / low probability sentences.
if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--jm_lambda", help="Parameter that controls " + \
                           "interpolation between unigram and bigram",
                           type=float, default=0.6, required=False)
    argparser.add_argument("--dir_alpha", help="Dirichlet parameter " + \
                           "for pseudocounts",
                           type=float, default=0.1, required=False)
    argparser.add_argument("--unk_cutoff", help="How many times must a word " + \
                           "be seen before it enters the vocabulary",
                           type=int, default=2, required=False)    
    argparser.add_argument("--katz_cutoff", help="Cutoff when to use Katz " + \
                           "backoff",
                           type=float, default=0.0, required=False)
    argparser.add_argument("--lm_type", help="Which smoothing technique to use",
                           type=str, default='mle', required=False)
    argparser.add_argument("--brown_limit", help="How many sentences to add " + \
                           "from Brown",
                           type=int, default=-1, required=False)
    argparser.add_argument("--kn_discount", help="Kneser-Ney discount parameter",
                           type=float, default=0.1, required=False)
    argparser.add_argument("--kn_concentration", help="Kneser-Ney concentration parameter",
                           type=float, default=1.0, required=False)
    argparser.add_argument("--method", help="Which LM method we use",
                           type=str, default='laplace', required=False)
    
    args = argparser.parse_args()    
    lm = BigramLanguageModel(kUNK_CUTOFF, jm_lambda=args.jm_lambda,
                             dirichlet_alpha=args.dir_alpha,
                             katz_cutoff=args.katz_cutoff,
                             kn_concentration=args.kn_concentration,
                             kn_discount=args.kn_discount)

    for ii in nltk.corpus.brown.sents():
        for jj in lm.tokenize(" ".join(ii)):
            lm.train_seen(lm._normalizer(jj))

    print("Done looking at all the words, finalizing vocabulary")
    lm.finalize()

    print('Vocabulary before adding unkown token')
    print(lm.get_vocab_len())


    sentence_count = 0
    for ii in nltk.corpus.brown.sents():
        sentence_count += 1
        lm.add_train(" ".join(ii))

        if args.brown_limit > 0 and sentence_count >= args.brown_limit:
            break


    print('Vocabulary after adding unkown token')
    print(lm.get_vocab_len())

    print('counts  of unkown token=', lm._unigram_dict[lm._unk_token])
    print('counts of word "the"=', lm._unigram_dict['the'])
    print('counts of word "directional"=', lm._unigram_dict['directional'])

    print("Trained language model with %i sentences from Brown corpus." % sentence_count)
    assert args.method in ['kneser_ney', 'mle', 'dirichlet', \
                           'jelinek_mercer', 'good_turing', 'laplace'], \
      "Invalid estimation method"

    # sent = input()
    # while sent:
    #     print("#".join(str(x) for x in lm.tokenize_and_censor(sent)))
    #     print(lm.perplexity(sent, getattr(lm, args.method)))
    #     sent = input()



    # """ 
    # Testing Treebank data set (getting the 10 most probable words)
    # i.e: (the 10 lowest preplexities)
    # """
    # print("Testing Treebank Corpus ........")
    # top_k = 10
    # top_k_list = [] # [(preplixty, index_in_corpus)]

    # sent_count = 0
    # for i in range(len(nltk.corpus.treebank.sents())):
    # # for i in range(500):
    #     sent_count +=1
    #     print('sent count: ', sent_count) 
    #     sent = " ".join(nltk.corpus.treebank.sents()[i])
    #     prep = lm.perplexity(sent, getattr(lm, args.method))


    #     if len(top_k_list) == 0:
    #         top_k_list.append((prep, i))

    #     elif (len(top_k_list) == top_k) and (prep < top_k_list[-1][0]):
    #         top_k_list.pop(-1) # removes the highest preplexity
    #         top_k_list.append((prep, i))
    #         top_k_list.sort()

    #     elif len(top_k_list) < top_k:
    #         top_k_list.append((prep, i))
    #         top_k_list.sort()


    # count = 0
    # for prep, i in top_k_list:
    #     count +=1
    #     sent = " ".join(nltk.corpus.treebank.sents()[i])
    #     print(f'({count})', prep, "  ->", sent)
    #     print("#".join(str(x) for x in lm.tokenize_and_censor(sent)))
    #     print('-----')



    sent = input()
    predicted_sent = sent
    while 1:
        word = lm.predict_word(predicted_sent, getattr(lm, args.method))
        predicted_sent += f' {word}'
        print(predicted_sent)
        print("#".join(str(x) for x in lm.tokenize_and_censor(predicted_sent)))
        print(lm.perplexity(predicted_sent, getattr(lm, args.method)))
        print('type "n22" to insert new sentences, or cotinue predicting')
        print('---------------------')
        sent = input()
        if (sent == 'n22'):
            predicted_sent = input()
        elif sent != '':
            predicted_sent += f' {sent}'

