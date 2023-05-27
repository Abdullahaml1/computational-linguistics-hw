import argparse
import torch
import torch.nn as nn
import numpy as np
import random

from torch.utils.data import Dataset
from torch.nn.utils import clip_grad_norm_
import json
import time
import nltk
import matplotlib.pyplot as plt

SEED = 1

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

kUNK = '<unk>'
kPAD = '<pad>'

# You don't need to change this funtion
def class_labels(data):
    class_to_i = {}
    i_to_class = {}
    i = 0
    for _, ans in data:
        if ans not in class_to_i.keys():
            class_to_i[ans] = i
            i_to_class[i] = ans
            i+=1
    return class_to_i, i_to_class

# You don't need to change this funtion
def load_data(filename, lim):
    """
    load the json file into data list
    param filename: josn faile as list of dictionaries
        i.e: [{'question_text': 'what a code !', 'label':1}, .....]
    """

    data = list()
    with open(filename) as json_data:
        if lim>0:
            # questions = json.load(json_data)["questions"][:lim]
            questions = json.load(json_data)[:lim] # list
        else:
            # questions = json.load(json_data)["questions"]
            questions = json.load(json_data) #list
        for q in questions:
            q_text = nltk.word_tokenize(q['question_text'])
            # q_text = nltk.word_tokenize(q['text'])
            #label = q['category']
            label = q['label']
            # label = q['page']
            # if label : # Error bypassing label: 0
            if label != None:
                data.append((q_text, label))
    return data

# You don't need to change this funtion
def load_words(exs):
    """
    vocabuary building
    Keyword arguments:
    exs: list of input questions-type pairs
    """

    words = set()
    word2ind = {kPAD: 0, kUNK: 1}
    ind2word = {0: kPAD, 1: kUNK}
    for q_text, _ in exs:
        for w in q_text:
            words.add(w)
    words = sorted(words)
    for w in words:
        idx = len(word2ind)
        word2ind[w] = idx
        ind2word[idx] = w
    words = [kPAD, kUNK] + words
    return words, word2ind, ind2word

def load_glove(weights_path):
    '''
    loading glove: vocab and embeddings
    :param weights_pat: (str) a .txt file with embeddings and vocab
    :return: (words, word_to_i, i_to_word, embeddings)
        words: a set of words in vocab len=vocab
        word_to_i: a (dict) {word: ind} len=vocab
        i_to_word: a (list) of words len=vocab
        embeddings: a tensor with shape=vocab x embed_dim
    '''
    # Unkown token in i_to_word[-1] <unk> in orignal
    # weighs, but after adding padding i_to_word[<unk>]=-2
    # i_to_word[<pad>] = -1
    i_to_word,embeddings = [],[]
    words = set()
    word_to_i = {}
    with open(weights_path,'rt') as fi:
        full_content = fi.read().strip().split('\n')
        # print(full_content[-1])
    for i in range(len(full_content)):
        word = full_content[i].split(' ')[0]
        embed = [float(val) for val in full_content[i].split(' ')[1:]]
        i_to_word.append(word)
        word_to_i[word] = i
        words.add(word)
        embeddings.append(embed)

    # adding padding token:
    i_to_word.append(kPAD)
    word_to_i[kPAD] = len(word_to_i)
    words.add(kPAD)
    embeddings.append(len(embeddings[0]) * [0.0]) 
    # print(embeddings[word_to_i[kPAD]][:10])

    return (words, word_to_i, i_to_word,
            torch.tensor(embeddings))



class QuestionDataset(Dataset):
    """
    Pytorch data class for questions
    """

    ###You don't need to change this funtion
    def __init__(self, examples, word2ind, num_classes, class2ind=None):
        self.questions = []
        self.labels = []

        for qq, ll in examples:
            self.questions.append(qq)
            self.labels.append(ll)
        
        # if type(self.labels[0])==str:
        for i in range(len(self.labels)):
            try:
                self.labels[i] = class2ind[self.labels[i]]
            except:
                self.labels[i] = num_classes
        self.word2ind = word2ind
    
    ###You don't need to change this funtion
    def __getitem__(self, index):
        return self.vectorize(self.questions[index], self.word2ind), \
          self.labels[index]
    
    ###You don't need to change this funtion
    def __len__(self):
        return len(self.questions)

    @staticmethod
    def vectorize(ex, word2ind):
        """
        vectorize a single example based on the word2ind dict. 
        Keyword arguments:
        exs: list of input questions-type pairs
        ex: tokenized question sentence (list)
        label: type of question sentence
        Output:  vectorized sentence(python list) and label(int)
        e.g. ['text', 'test', 'is', 'fun'] -> [0, 2, 3, 4]
        """

        vec_text = [0] * len(ex)

        #### modify the code to vectorize the question text
        #### You should consider the out of vocab(OOV) cases
        #### question_text is already tokenized    
        ####Your code here
        for ii in range(len(ex)):
            if ex[ii] not in word2ind:
                vec_text[ii] = word2ind[kUNK]
            else:
                vec_text[ii] = word2ind[ex[ii]]
        return vec_text

    
###You don't need to change this funtion

def batchify(batch):
    """
    Gather a batch of individual examples into one batch, 
    which includes the question text, question length and labels 
    Keyword arguments:
    batch: list of outputs from vectorize function
    :return: dictionary -> {'text': tensor(batch x max(question_len)),
                            'len;: tensor(batch x 1),
                            'label': tensor(batch x 1)
    """

    question_len = list()
    label_list = list()
    for ex in batch:
        question_len.append(len(ex[0]))
        label_list.append(ex[1])

    target_labels = torch.LongTensor(label_list)
    # initializing the input vector and pad the rest with zeros
    x1 = torch.LongTensor(len(question_len), max(question_len)).zero_()
    for i in range(len(question_len)):
        question_text = batch[i][0]
        vec = torch.LongTensor(question_text)
        x1[i, :len(question_text)].copy_(vec)
    q_batch = {'text': x1, 'len': torch.FloatTensor(question_len), 'labels': target_labels}
    return q_batch



def evaluate(data_loader, model, loss_fun, device):
    """
    evaluate the current model, get the accuracy for dev/test set
    Keyword arguments:
    data_loader: pytorch build-in data loader output
    model: model to be evaluated
    device: cpu of gpu
    """

    model.eval()
    num_examples = 0
    error = 0

    total_loss = 0.0
    num_examples = 0
    with torch.no_grad():
        for idx, batch in enumerate(data_loader):
            question_text = batch['text'].to(device)
            question_len = batch['len']
            labels = batch['labels']
    
            ####Your code here

            logits = model(question_text, question_len) # shape [batch x num_classes]
            top_n, top_i = logits.topk(1)
            num_examples += question_text.size(0)
            error += torch.nonzero(top_i.squeeze() - torch.LongTensor(labels)).size(0)

            # Loss
            total_loss += loss_fun(logits, labels).item()

        # Accuracy
        accuracy = 1 - error / num_examples
        avg_loss = total_loss/num_examples
    # print(f'Dev accuracy={accuracy:f}, Dev average Loss={avg_loss:f}')
    return accuracy, avg_loss



def train(args, model, train_data_loader, dev_data_loader, accuracy, device):
    """
    Train the current model
    Keyword arguments:
    args: arguments 
    model: model to be trained
    train_data_loader: pytorch build-in data loader output for training examples
    dev_data_loader: pytorch build-in data loader output for dev examples
    accuracy: previous best accuracy
    device: cpu of gpu
    """

    model.train()
    optimizer = torch.optim.Adamax(model.parameters())
    criterion = nn.CrossEntropyLoss()
    print_loss_total = 0
    epoch_loss_total = 0
    start = time.time()

    #### modify the following code to complete the training funtion
    train_loss_list= []
    train_acc_list = []
    train_error = 0
    num_train_examples = 0

    dev_loss_list = []
    dev_acc_list = []
    new_best = False

    for idx, batch in enumerate(train_data_loader):
        question_text = batch['text'].to(device)
        question_len = batch['len']
        labels = batch['labels']

        #### Your code here
        question_text.to(device)
        question_len.to(device)
        labels.to(device)


        # forward
        pred = model(question_text, question_len)

        # computing loss
        loss = criterion(pred, labels)


        # computing gradient
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        

        clip_grad_norm_(model.parameters(), args.grad_clipping) 
        print_loss_total += loss.data.numpy()
        epoch_loss_total += loss.data.numpy()


        # Logging train accuracy
        _, top_i = pred.topk(1, dim=1)
        train_error = torch.nonzero(labels.squeeze() - top_i.squeeze()).shape[0]
        num_train_examples += question_text.shape[0]

        # Reaching checkpoint
        if idx % args.checkpoint == 0 and idx > 0:


            # Applying Devset
            dev_curr_accuracy, dev_curr_loss = evaluate(dev_data_loader, model,
                    nn.CrossEntropyLoss(), device)

            # Logging Train Loss and Accuracy
            # print_loss_avg = print_loss_total / args.checkpoint
            print_loss_avg = print_loss_total / num_train_examples
            train_loss_list.append(print_loss_avg.item())

            train_acc = 1- train_error/num_train_examples
            train_acc_list.append(train_acc)

            num_train_examples = 0
            train_error =0
            print_loss_total = 0

            # Logging Dev Loss and Accuracy
            dev_acc_list.append(dev_curr_accuracy)
            dev_loss_list.append(dev_curr_loss)

            if accuracy < dev_curr_accuracy:
                print('Saving Model ............')
                torch.save(model, args.save_model)

                new_best = True
                accuracy = dev_curr_accuracy


            # print('number of steps: %d, loss: %.5f time: %.5f' % (idx, print_loss_avg, time.time()- start))
            print(f'# of steps={idx}, Avg Train Loss={print_loss_avg:f}'+ 
                    f', Avg Dev Loss={dev_curr_loss:f}, Train Acc={train_acc:f}'+ 
                    f', Dev Acc={dev_curr_accuracy:f}, Time: {time.time()-start:f}') 

    return {'dev_best_acc': accuracy,
            'new_best': new_best,
            'train_acc_epoch': train_acc_list,
            'train_loss_epoch': train_loss_list,
            'dev_acc_epoch': dev_acc_list,
            'dev_loss_epoch': dev_loss_list}





class DanModel(nn.Module):
    """High level model that handles intializing the underlying network
    architecture, saving, updating examples, and predicting examples.
    """

    #### You don't need to change the parameters for the model for passing tests, might need to tinker to improve performance/handle
    #### pretrained word embeddings/for your project code.


    def __init__(self, n_classes, vocab_size, emb_dim=50,
                 n_hidden_units=50, nn_dropout=.5, padding_idx=0,
                 glove_embeddings=None):
        super(DanModel, self).__init__()
        self.n_classes = n_classes
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.n_hidden_units = n_hidden_units
        self.nn_dropout = nn_dropout

        # Embedding Layer
        if glove_embeddings != None :
            self.embeddings = nn.Embedding.from_pretrained(glove_embeddings,
                                    padding_idx=padding_idx)
            self.emb_dim = self.embeddings.embedding_dim
            # self.emb_dim = glove_embeddings.shape[1]
        else:
            self.embeddings = nn.Embedding(self.vocab_size, self.emb_dim,
                    padding_idx=padding_idx)

        self.linear1 = nn.Linear(self.emb_dim, n_hidden_units)
        self.linear2 = nn.Linear(n_hidden_units, n_classes)
        self._softmax = nn.Softmax(dim=1) # dim == axis

        # Create the actual prediction framework for the DAN classifier.

        # You'll need combine the two linear layers together, probably
        # with the Sequential function.  The first linear layer takes
        # word embeddings into the representation space, and the
        # second linear layer makes the final prediction.  Other
        # layers / functions to consider are Dropout, ReLU. 
        # For test cases, the network we consider is - linear1 -> ReLU() -> Dropout(0.5) -> linear2

        #### Your code here
        self.linear_stack = torch.nn.Sequential(self.linear1,
                                                torch.nn.ReLU(),
                                                torch.nn.Dropout(self.nn_dropout),
                                                self.linear2)

        
        
       
    def forward(self, input_text, text_len, is_prob=False):
        """
        Model forward pass, returns the logits of the predictions.
        
        Keyword arguments:
        input_text : vectorized question text  [batch x questin_list]
        text_len : batch * 1, text length for each question
        is_prob: if True, output the softmax of last layer
        """

        # logits = torch.LongTensor([0.0] * self.n_classes)

        # Complete the forward funtion.  First look up the word embeddings.
        embd = self.embeddings(input_text) #batch x seq_len x embed_len
        
        # Then average them 
        embd = embd.sum(axis=1)/text_len.reshape([-1, 1]) # batch x embed_len
        
        # Before feeding them through the network
        logits = self.linear_stack(embd) # batch x label_len
        

        if is_prob:
            logits = self._softmax(logits)

        return logits

def show_error_samples(data_loader, model, loss_fun,
        ind2word_arr, device):
    """
    evaluate the current model, get the accuracy for dev/test set
    Keyword arguments:
    data_loader: pytorch build-in data loader output
    model: model to be evaluated
    device: cpu of gpu
    """

    model.eval()
    num_examples = 0
    error = 0

    total_loss = 0.0
    num_examples = 0
    with torch.no_grad():
        for idx, batch in enumerate(data_loader):
            question_text = batch['text'].to(device)
            question_len = batch['len']
            labels = batch['labels']
    
            ####Your code here

            logits = model(question_text, question_len) # shape [batch x num_classes]
            _, top_i = logits.topk(1)
            num_examples += question_text.size(0)
            error += torch.nonzero(top_i.squeeze() - torch.LongTensor(labels)).size(0)

            ## Error samples
            error_logits = logits[top_i.squeeze() != torch.LongTensor(labels)]
            error_samples_text = question_text[top_i.squeeze() != torch.LongTensor(labels)]
            error_samples_labels = labels[top_i.squeeze() != torch.LongTensor(labels)]

            print(error_samples_text.shape)
            print(error_samples_labels.shape)

            for i in range(len(error_samples_text)):
                print(ind2word_arr[error_samples_text[i]])


            # Loss
            total_loss += loss_fun(logits, labels).item()

        # Accuracy
        accuracy = 1 - error / num_examples
        avg_loss = total_loss/num_examples
    # print(f'Dev accuracy={accuracy:f}, Dev average Loss={avg_loss:f}')
    return accuracy, avg_loss


''' ploting function '''
def plot_model(train, test, num_epochs):
    '''
    plots a given train and test data
    :param ax: matplotlib ax
    :param title: str
    :param train: train list
    :param test: test list
    :param test_point: number
    '''

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    def plot_val(ax, title, train, test, num_epochs=num_epochs):
        # x = np.arange(1, len(test)+1, 1) * (num_epochs / len(test))
        x = np.arange(len(test)) * ((num_epochs-1) / (len(test)-1))
        ax.plot(x, train, label='train', color='r')
        ax.plot(x, test, label='dev', color='b')
        # ax.plot([len(test)], [test_point], 'g*')
        # ax.annotate(f"test {title}={test_point:.3f}", xy=(len(test), test_point), xytext=(len(test)-1, test_point-.05))
        ax.set_xlabel('Epochs')
        ax.legend()
        ax.set_title(title)
        ax.grid()

    plot_val(ax1, 'Accuracy', train['accuracy'], test['accuracy']) 
    plot_val(ax2, 'Loss', train['loss'], test['loss']) 
    return fig, (ax1, ax2)



# You basically do not need to modify the below code 
# But you may need to add funtions to support error analysis 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Question Type')
    parser.add_argument('--no-cuda', action='store_true', default=True)
    parser.add_argument('--train-file', type=str, default='data/question_train_cl1.json')
    parser.add_argument('--dev-file', type=str, default='data/question_dev_cl1.json')
    parser.add_argument('--test-file', type=str, default='data/question_test_cl1.json')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--num-epochs', type=int, default=20)
    parser.add_argument('--grad-clipping', type=int, default=5)
    parser.add_argument('--resume', action='store_true', default=False)
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--save-model', type=str, default='q_type.pt')
    parser.add_argument('--load-model', type=str, default='q_type.pt')
    parser.add_argument("--limit", help="Number of training documents", type=int, default=-1, required=False)
    parser.add_argument('--checkpoint', type=int, default=21)
    parser.add_argument("--num-workers", help="Number of workers", type=int, default=4, required=False)
    parser.add_argument('--use-glove', action='store_true', help='Wather to use glove or not', default=False)
    parser.add_argument("--glove-weights", help="Path to glove weights", type=str, default='', required=False)
    parser.add_argument('--show-dev-error-samples', action='store_true', help='Print Error Dev samples', default=False)

    args = parser.parse_args()
    #### check if using gpu is available
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")

    # model name
    if args.use_glove and (not args.show_dev_error_samples):
        args.save_model= f'glove_{args.glove_weights.split(".")[-2]}_{args.save_model}'
    else:
        args.save_model = f'normal_50d_{args.save_model}'

    ### Load data
    train_exs = load_data(args.train_file, args.limit)
    dev_exs = load_data(args.dev_file, -1)
    test_exs = load_data(args.test_file, -1)

    ### Create vocab
    if args.use_glove:
        print('Loading Glove Embeddings.......')
        voc, word2ind, ind2word, glove_embeddings = load_glove(args.glove_weights)
    else:
        voc, word2ind, ind2word = load_words(train_exs)

    #get num_classes from training + dev examples - this can then also be used as int value for those test class labels not seen in training+dev.
    num_classes = len(list(set([ex[1] for ex in train_exs+dev_exs])))
    print('Number of Classes=', num_classes)

    #get class to int mapping
    class2ind, ind2class = class_labels(train_exs + dev_exs)  

    if args.test:
        model = torch.load(args.load_model)
        #### Load batchifed dataset
        test_dataset = QuestionDataset(test_exs, word2ind, num_classes, class2ind)
        test_sampler = torch.utils.data.sampler.SequentialSampler(test_dataset)
        test_loader = torch.utils.data.DataLoader(test_dataset,
                                                batch_size=args.batch_size,
                                                sampler=test_sampler,
                                                num_workers=args.num_workers,
                                                collate_fn=batchify)
        evaluate(test_loader, model,nn.CrossEntropyLoss, device)

    # show Error Dev Samples
    elif args.show_dev_error_samples:
        model = torch.load(args.load_model)
        model.to(device)

        dev_dataset = QuestionDataset(dev_exs, word2ind, num_classes, class2ind)
        dev_sampler = torch.utils.data.sampler.SequentialSampler(dev_dataset)
        dev_loader = torch.utils.data.DataLoader(dev_dataset, batch_size=args.batch_size,
                                               sampler=dev_sampler,
                                               num_workers=args.num_workers,
                                               collate_fn=batchify)
        # Applying Devset
        dev_acc, dev_loss = show_error_samples(dev_loader, model,
                        nn.CrossEntropyLoss(), np.array(ind2word), device)
        print(f'Dev acc={dev_acc:f}, dev_error={dev_loss:f}')

    else:
        if args.resume:
            print('Resuming.....')
            model = torch.load(args.load_model)
        else:
            if args.use_glove:
                model = DanModel(num_classes, len(voc),
                        glove_embeddings=glove_embeddings,
                        padding_idx=word2ind[kPAD])
            else:
                model = DanModel(num_classes, len(voc))

            model.to(device)
        print(model)
        #### Load batchifed dataset
        train_dataset = QuestionDataset(train_exs, word2ind, num_classes, class2ind)
        train_sampler = torch.utils.data.sampler.RandomSampler(train_dataset)

        ''' Debug Print '''
        # train_sample = next(iter(train_dataset))
        # print('Debug DebugDebugDebugDebug----------------')
        # print(train_sample[0])
        # print(train_sample[1])

        dev_dataset = QuestionDataset(dev_exs, word2ind, num_classes, class2ind)
        dev_sampler = torch.utils.data.sampler.SequentialSampler(dev_dataset)
        dev_loader = torch.utils.data.DataLoader(dev_dataset, batch_size=args.batch_size,
                                               sampler=dev_sampler,
                                               num_workers=args.num_workers,
                                               collate_fn=batchify)

        ''' Debug Print '''
        # train_loader_debug = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
        #                                       sampler=train_sampler,
        #                                       num_workers=args.num_workers,
        #                                       collate_fn=batchify)
        # train_sample = next(iter(train_loader_debug))
        # print('Debug DebugDebugDebugDebug----------------')
        # print(train_sample['text'].shape) # [batch x 60]
        # print(train_sample['len'].shape) # [batch x 60]
        # print(train_sample['labels'].shape) # [batch]
        # print(model(train_sample['text'],train_sample['len']).shape)

        ''' Training LOOOP'''
        accuracy = 0
        train_acc_list = []
        train_loss_list =[]
        dev_acc_list = []
        dev_loss_list = []
        best_epoch = 0
        for epoch in range(args.num_epochs):
            print('start epoch %d' % epoch)
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                               sampler=train_sampler,
                                               num_workers=args.num_workers,
                                               collate_fn=batchify)
            log_dict = train(args, model, train_loader, dev_loader, accuracy, device)
            if log_dict['new_best']:
                best_epoch = epoch

            accuracy = log_dict['dev_best_acc']
            train_acc_list.append(log_dict['train_acc_epoch'])
            train_loss_list.append(log_dict['train_loss_epoch'])
            dev_acc_list.append(log_dict['dev_acc_epoch'])
            dev_loss_list.append(log_dict['dev_loss_epoch'])
            print('----------------------------------------------------------')
            print()

        # Plotting
        train_dict = {'accuracy': np.array(train_acc_list).reshape([-1,]),
                        'loss': np.array(train_loss_list).reshape([-1,])}
        dev_dict = {'accuracy': np.array(dev_acc_list).reshape([-1]),
                    'loss': np.array(dev_loss_list).reshape([-1])}
        fig, (ax1, ax2)= plot_model(train_dict, dev_dict, num_epochs=args.num_epochs)
        plt.show()
        if args.use_glove:
            fig.savefig(f'./glove_{args.glove_weights.split(".")[-2]}_plot.png')
        else:
            fig.savefig(f'./normal_50d_plot.png')

        print(f'Best Epoch = {best_epoch}')
        print('start testing:\n')
        test_dataset = QuestionDataset(test_exs, word2ind, num_classes, class2ind)
        test_sampler = torch.utils.data.sampler.SequentialSampler(test_dataset)
        test_loader = torch.utils.data.DataLoader(test_dataset,
                                                batch_size=args.batch_size,
                                                sampler=test_sampler,
                                                num_workers=args.num_workers,
                                                collate_fn=batchify)
        test_acc, test_loss = evaluate(test_loader, model, nn.CrossEntropyLoss(), device)
        print(f'Test Acc={test_acc}, Test Loss={test_loss}')
