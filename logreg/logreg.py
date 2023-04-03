
import random
import numpy as np
from math import exp, log
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib_inline

import argparse

SEED = 42
kSEED = 1701
kBIAS = "BIAS_CONSTANT"

SMALL_NUMBER = 1e-8

np.random.seed(SEED)
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
    return -y/(p + SMALL_NUMBER) - log2(p) + \
        (1-y)/(1- p + SMALL_NUMBER) + log2(1-p)
    # if (y==1):
    #     return -1/(p + SMALL_NUMBER) - log2(p)
    # else:
    #     return 1/(1- p + SMALL_NUMBER) + log2(1-p)


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




class ExamplesDataset:
    '''
    class to represent dataset (pool of Example objects)
    '''
    def __init__(self, positive, negative, vocab, test_proportion=.1):
        """
        :param positive: Positive examples file
        :param negative: Negative examples file
        :param vocab: A list of vocabulary words file
        :param test_proprotion: How much of the data should be reserved for test (int)
        """
        self.train_examples_list =[]
        self.test_examples_list =[]
        self.vocab_list =[]

        # shapes are not fake (just to tell you the are numby arrays
        self.train_features_arr = np.empty(shape=(100,1000)) 
        self.test_features_arr = np.empty(shape=(100,1000)) 

        # a dict with {1:'positive list', 0'negative list'}
        self.examples_dict = {1:[], 0:[]}

        # reeagin dataset
        self.read_dataset(positive, negative, vocab, test_proportion=.1)


    def read_dataset(self, positive, negative, vocab, test_proportion=.1):
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
                self.examples_dict[label].append(ex)
                if random.random() <= test_proportion:
                    test.append(ex)
                else:
                    train.append(ex)
    
        # Shuffle the data so that we don't have order effects
        random.shuffle(train)
        random.shuffle(test)

        self.train_examples_list = train
        self.test_examples_list = test
        self.vocab_list = vocab

        # converting to arrays
        self.train_features_arr = np.empty(shape=(len(train), len(vocab)))
        for i in range(len(train)):
            self.train_features_arr[i] = train[i].x
            
        self.test_features_arr = np.empty(shape=(len(test), len(vocab)))
        for i in range(len(test)):
            self.test_features_arr[i] = test[i].x

    def normalize(self):
        """
        normalizing the dataset 
        """
        mean = np.mean(self.train_features_arr, axis=0)
        std = np.std(self.train_features_arr, axis=0)
        std[std < SMALL_NUMBER] = 1 # avoiding devide by zero errory
        self.train_features_arr -= mean
        self.train_features_arr /= std

        mean = np.mean(self.test_features_arr, axis=0)
        std = np.std(self.test_features_arr, axis=0)
        std[std < SMALL_NUMBER] = 1 # avoiding devide by zero errory
        self.test_features_arr -= mean
        self.test_features_arr /= std

        for i in range(len(self.train_features_arr)):
            self.train_examples_list[i].x = self.train_features_arr[i]

        for i in range(len(self.test_features_arr)):
            self.test_examples_list[i].x = self.test_features_arr[i]



    def get_examples(self):
        '''
        get examples of the dataset
        :return: a tuple of (train, test, vocab)
        train: list of Example object
        test: list of Example object
        vocab: list of words
        '''

        return (self.train_examples_list, 
                self.test_examples_list,
                self.vocab_list)

    def get_positive_negative_examples(self):
        '''
        :return: a tuple (positive_list,
                        negative_list,
                        voab list of words)
        '''
        return (self.examples_dict[1],
                self.examples_dict[0],
                self.vocab_list)




class LogReg:
    def __init__(self, num_features, mu, step,
            loss_func=cross_entropy_loss,
            loss_deriv_func=cross_entropy_loss_derivative):
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
        self.loss_func = loss_func
        self.loss_deriv_func = loss_deriv_func
        self.loss = 0.0 # accumalate loss

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
        :return: A tuple of (log probability, accuracy, loss)
        """

        logprob = 0.0
        num_right = 0
        loss = 0.0
        for ii in examples:
            p = self.forward(ii)
            loss += self.loss_func(p, ii.y)
            if ii.y == 1:
                logprob += log(p)
            else:
                logprob += log(1.0 - p)

            if self.mu > 0:
                logprob -= self.mu * np.sum(self.beta ** 2)

            # Get accuracy
            if abs(ii.y - p) < 0.5:
                num_right += 1

        return logprob, float(num_right) / float(len(examples)), float(loss/len(examples))

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

        loss_deriv = self.loss_deriv_func(p, train_example.y)
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


''' ploting function '''
def plot_test_val(ax, title, train, test, test_point):
    '''
    plots a given train and test data
    :param ax: matplotlib ax
    :param title: str
    :param train: train list
    :param test: test list
    :param test_point: number
    '''
    x = np.arange(1, len(test) +1, 1)
    ax.plot(x, train, label='train', color='r')
    ax.plot(x, test, label='test', color='b')
    ax.plot([len(test)], [test_point], 'g*')
    ax.annotate(f"test {title}={test_point:.3f}", xy=(len(test), test_point), xytext=(len(test)-1, test_point-.05))
    ax.set_xlabel('Epochs')
    ax.legend()
    ax.set_title(title)
    ax.grid()

    
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
    argparser.add_argument("--early_stop", help="Early stop of test loss increased | {yes|no}",
                           type=str, default='no')
    argparser.add_argument("--normalize", help="normalize the dataset | {yes|no}",
                           type=str, default='no')
    argparser.add_argument("--log", help="display statics or not | {yes:no}",
                           type=str, default='yes')
    argparser.add_argument("--log_step", help="rate to print single epoch log",
                           type=int, default=100)
    argparser.add_argument("--plot_name", help="plot name",
                           type=str, default='plot')

    args = argparser.parse_args()

    '''Reading dataset'''
    dataset = ExamplesDataset(args.positive, args.negative, args.vocab)
    if args.normalize=='yes':
        dataset.normalize()
    train, test, vocab = dataset.get_examples()
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
    train_loss_list = []
    test_loss_list =[]

    train_acc_list=[]
    test_acc_list=[]
    _, _,last_test_loss = lr.progress(test)
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

            if update_number % args.log_step == 1:
                train_lp, train_acc, train_loss = lr.progress(train)
                ho_lp, ho_acc, test_loss = lr.progress(test) # h for hypotheses
                if args.log=='yes' :
                    print("    Update %i\tTP %f\tHP %f\tTA %f\tHA %f" %
                        (update_number, train_lp, ho_lp, train_acc, ho_acc))
        
        '''recording losses for ploting'''
        train_loss_list.append(train_loss)
        test_loss_list.append(test_loss)
        train_acc_list.append(train_acc)
        test_acc_list.append(ho_acc)

        """ Early Stoping"""
        if args.early_stop=='yes':
            if test_loss > last_test_loss:
                print('Early Stop')
                break
            last_test_loss = test_loss

        ''' displaying loss'''
        print(f"Train logP={train_lp:f} Test logP={ho_lp:f} Train Acc={train_acc:f} Test Acc={ho_acc:f} " +
                f'TrainLoss={train_loss:f} TestLoss={test_loss:f}')
        print('----------------------------')

    # Final update with empty example
    lr.finalize_lazy(update_number)
    print(f"Train logP={train_lp:f} Test logP={ho_lp:f} Train Acc={train_acc:f} Test Acc={ho_acc:f} " +
            f'TrainLoss={train_loss:f} TestLoss={test_loss:f}')


    '''ploting results'''
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    plot_test_val(ax1, 'Loss', train_loss_list, test_loss_list, test_loss)
    plot_test_val(ax2, 'Accuracy', train_acc_list, test_acc_list, ho_acc)
    plt.show()
    fig.savefig(f'figures/{args.plot_name}.png')

