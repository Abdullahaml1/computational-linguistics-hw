
import random
import numpy as np
from math import exp, log
from collections import defaultdict

import argparse

kSEED = 1701
kBIAS = "BIAS_CONSTANT"

SMALL_NUMBER = 1e-8

random.seed(kSEED)


def sigmoid(score, threshold=20.0):
    """
    Prevent overflow of exp by capping activation at 20.
    :param score: A real valued number to convert into a number between 0 and 1
    """

    if abs(score) > threshold:
        score = threshold * np.sign(score)

    activation = exp(score)
    return activation / (1.0 + activation)

def sigmoid_derivative(sig_output):
    """
    ;param sig_output: the output of sigmoid function
    :return: the dervative of sigmoid
    """
    return sig_output * (1-sig_output)

def log2(x):
    """
    :return: log with base 2
    """
    return np.log(x+SMALL_NUMBER)/np.log(2)

def cross_entropy_loss(p, y):
    """
    :param p: predicte value
    :param y: true value
    :return: corssEntropy loss
    """
    if (y==1):
        return -log2(p)
    else:
        return -log2(1-p)


def cross_entropy_loss_derivative(p, y):
    """
    :param p: predicte value
    :param y: true value
    :return: corssEntropy loss dervative
    """
    if (y==1):
        return -1/(p + SMALL_NUMBER) - log2(p)
    else:
        return -1/(1- p + SMALL_NUMBER) - log2(1-p)


class Example:
    """
    Class to represent a logistic regression example
    """
    def __init__(self, label, words, vocab, df):
        """
        Create a new example
        :param label: The label (0 / 1) of the example
        :param words: The words in a list of "word:count" format
        :param vocab: The vocabulary to use as features (list)
        """
        self.nonzero = {vocab.index(kBIAS): 1}
        self.y = label
        self.x = np.zeros(len(vocab))
        for word, count in [x.split(":") for x in words]:
            if word in vocab:
                assert word != kBIAS, "Bias can't actually appear in document"
                self.x[vocab.index(word)] += float(count)
                self.nonzero[vocab.index(word)] = word
        self.x[0] = 1 # the bias


class LogReg:
    def __init__(self, num_features, mu, step):
        """
        Create a logistic regression classifier
        :param num_features: The number of features (including bias)
        :param mu: Regularization parameter (for extra credit)
        :param step: A function that takes the iteration as an argument (the default is a constant value) -> (learing rate)
        """

        self.dimension = num_features
        self.beta = np.random.randn(num_features) # weights
        self.mu = mu
        self.step = step # learning rate
        self.last_update = np.zeros(num_features)

        assert self.mu >= 0, "Regularization parameter must be non-negative"

    def forward(self, example):
        """
        returns the forward probability for a SINGLE example
        :param example: object of type Example
        :return: the probabiltiy
        """
        return sigmoid(np.dot(self.beta, example.x))

    def progress(self, examples):
        """
        Given a set of examples, compute the probability and accuracy
        :param examples: The dataset to score
        :return: A tuple of (log probability, accuracy)
        """

        logprob = 0.0
        num_right = 0
        for ii in examples:
            p = self.forward(ii)
            if ii.y == 1:
                logprob += log(p)
            else:
                logprob += log(1.0 - p)

            if self.mu > 0:
                logprob -= self.mu * np.sum(self.beta ** 2)

            # Get accuracy
            if abs(ii.y - p) < 0.5:
                num_right += 1

        return logprob, float(num_right) / float(len(examples))

    def sg_update(self, train_example, iteration,
                  lazy=False, use_tfidf=False):
        """
        Compute a stochastic gradient update to improve the log likelihood.
        :param train_example: The example to take the gradient with respect to
        :param iteration: The current iteration (an integer)
        :param use_tfidf: A boolean to switch between the raw data and the tfidf representation
        :return: Return the new value of the regression coefficients
        """
        p = self.forward(train_example)

        loss_deriv = cross_entropy_loss_derivative(p, train_example.y)
        sig_deriv = sigmoid_derivative(p)

        grad = loss_deriv * sig_deriv * train_example.x

        self.beta -= self.step(1) * grad # self.step is a lamabda function

        return self.beta

    def finalize_lazy(self, iteration):
        """
        After going through all normal updates, apply regularization to
        all variables that need it.
        Only implement this function if you do the extra credit.
        """


        return self.beta

def read_dataset(positive, negative, vocab, test_proportion=.1):
    """
    Reads in a text dataset with a given vocabulary
    :param positive: Positive examples
    :param negative: Negative examples
    :param vocab: A list of vocabulary words
    :param test_proprotion: How much of the data should be reserved for test
    """
    df = [float(x.split("\t")[1]) for x in open(vocab, 'r') if '\t' in x] # count of words in the vocab
    vocab = [x.split("\t")[0] for x in open(vocab, 'r') if '\t' in x] # list of words in vocab
    assert vocab[0] == kBIAS, \
        "First vocab word must be bias term (was %s)" % vocab[0]

    train = []
    test = []
    for label, input in [(1, positive), (0, negative)]:
        for line in open(input):
            ex = Example(label, line.split(), vocab, df)
            if random.random() <= test_proportion:
                test.append(ex)
            else:
                train.append(ex)

    # Shuffle the data so that we don't have order effects
    random.shuffle(train)
    random.shuffle(test)

    return train, test, vocab

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--mu", help="Weight of L2 regression",
                           type=float, default=0.0, required=False)
    argparser.add_argument("--step", help="Initial SG step size (learning rate)",
                           type=float, default=0.1, required=False)
    argparser.add_argument("--positive", help="Positive class",
                           type=str, default="data/positive", required=False)
    argparser.add_argument("--negative", help="Negative class",
                           type=str, default="data/negative", required=False)
    argparser.add_argument("--vocab", help="Vocabulary that can be features",
                           type=str, default="data/vocab", required=False)
    argparser.add_argument("--passes", help="Number of passes through train",
                           type=int, default=1, required=False)
    argparser.add_argument("--ec", help="Extra credit option (df, lazy, or rate)",
                           type=str, default="")
    argparser.add_argument("--log", help="display statics or not",
                           type=str, default='True')

    args = argparser.parse_args()
    train, test, vocab = read_dataset(args.positive, args.negative, args.vocab)

    print("Read in %i train and %i test" % (len(train), len(test)))

    # Initialize model
    if args.ec != "rate":
        lr = LogReg(len(vocab), args.mu, lambda x: args.step)
    else:
        # Modify this code if you do learning rate extra credit
        raise NotImplementedError

    """
    MAIN LOOP
    """
    # Iterations
    for pp in range(args.passes):
        print(f'Epoch:{pp+1}')
        update_number = 0
        for ii in train:
            update_number += 1
            # Do we use extra credit option
            if args.ec == "df":
                lr.sg_update(ii, update_number, use_tfidf=True)
            elif args.ec == "lazy":
                lr.sg_update(ii, update_number, lazy=True)
            else:
                lr.sg_update(ii, update_number)

            if update_number % 30 == 1:
                train_lp, train_acc = lr.progress(train)
                ho_lp, ho_acc = lr.progress(test) # h for hypotheses
                if args.log == 'True':
                    print("    Update %i\tTP %f\tHP %f\tTA %f\tHA %f" %
                        (update_number, train_lp, ho_lp, train_acc, ho_acc))

    # Final update with empty example
    lr.finalize_lazy(update_number)
    print("Update %i\tTP %f\tHP %f\tTA %f\tHA %f" %
          (update_number, train_lp, ho_lp, train_acc, ho_acc))
