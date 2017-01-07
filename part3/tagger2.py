import numpy as np
import dynet as dy
import random
import sys
import re

STUDENT={'name': 'Yaniv Sheena',
         'ID': '308446764'}


# NER task files paths
NER_TRAIN_FILE = 'data/ner/train'
NER_DEV_FILE   = 'data/ner/dev'
# POS task files paths
POS_TRAIN_FILE = 'data/pos/train'
POS_DEV_FILE   = 'data/pos/dev'


def get_words_vec(vocab_file, wordsvec_file):
  with open(vocab_file, 'r') as f:
      data_lines = f.readlines()

  # get words from the vocabulary
  words = [w.strip() for w in data_lines]
  w2ind = {w: i for i,w in enumerate(words)}

  # get all pre-trained words vector (E)
  words_vec = np.loadtxt(wordsvec_file)

  return w2ind, words_vec

def get_examples_set(examples_file):
    examples = []
    # the order is important here - we get a sequence
    for line in open(examples_file, 'r'):
        ex = tuple(line.strip().split())
        if len(ex) == 2:
            examples.append(ex)

    return examples


class dynet_model:
    def __init__(self, indexed_vocab, indexed_labels, init_mapping, external_E, hid_dim=100, emb_dim=50):

        self.indexed_vocab = indexed_vocab
        self.indexed_labels = indexed_labels
        # reverse dict - from index to label
        self.i2l = {i: label for label, i in indexed_labels.iteritems()}

        self.vocab_size = len(indexed_vocab)
        self.out_dim = len(indexed_labels)            

        # define the parameters
        self.model = dy.Model()

        # first layer params
        self.pW1 = self.model.add_parameters((hid_dim, 5*emb_dim))
        self.pb1 = self.model.add_parameters(hid_dim)

        # hidden layer params
        self.pW2 = self.model.add_parameters((self.out_dim, hid_dim))
        self.pb2 = self.model.add_parameters(self.out_dim)

        # word embedding - E 
        self.E = self.model.add_lookup_parameters((self.vocab_size,emb_dim))

        # init word embedding matrix (E) with the given pre-trained external words
        # the last rows of E will be still randomly init (see above) due to the addition of
        # extra special words to the vocabulary and not to the external E
        self.E.init_from_array(external_E)

        # init rows of E with other rows of E according to init_mapping
        # (Init word with the vectors for their lower case version)
        for w, i in init_mapping.iteritems():
            self.E.init_row(indexed_vocab[w], self.E[i].value())

        # some internal inits
        self.dev_accuracies = []
        self.dev_losses = []

    def predict_labels(self, w_sequence):
        x = self.encode_seq(w_sequence)
        h = self.layer1(x)
        y = self.layer2(h)
        return dy.softmax(y)

    def layer1(self, x):
        W = dy.parameter(self.pW1)
        b = dy.parameter(self.pb1)
        return dy.tanh(W*x+b)

    def layer2(self, x):
        W = dy.parameter(self.pW2)
        b = dy.parameter(self.pb2)
        return W*x+b

    def encode_seq(self, w_sequence):
        ''' Concatenating word vectors within w_sequence '''
        indexes = [self.indexed_vocab[w] for w in w_sequence]
        embs = [self.E[idx] for idx in indexes]
        return dy.concatenate(embs) 

    def do_loss(self, probs, label):
        label = self.indexed_labels[label]
        return -dy.log(dy.pick(probs,label))

    def classify(self, w_sequence, label):
        dy.renew_cg()
        probs = self.predict_labels(w_sequence)
        vals = probs.npvalue()
        return np.argmax(vals), -np.log(vals[label])

    def predict(self, w_sequence):
        ''' classify without the loss '''
        prediction, _ = self.classify(w_sequence, 0)
        return prediction

    def predict_blind_test(self, test_data):
        ''' gets a function that predicts tag, a test file path and a dict converting
            numric tags to their corresponding string '''
        print 'Predicting blind test'
        predictions = []
        for seq in test_data:
            predictions.append(self.i2l[self.predict(seq)])

        return predictions

    def write_dev_accuracies_to_file(self):
        print 'Writing dev results to files..'
        np.savetxt("dev_accuracies.txt", np.array(self.dev_accuracies))
        np.savetxt("dev_loss.txt", np.array(self.dev_losses))

    def train_model(self, train_data, dev_data, learning_rate=0.01, max_iterations=100):
        trainer = dy.SimpleSGDTrainer(self.model)
        best_dev_loss = 1e3
        best_iter = 0
        print 'Start training the model..'
        for ITER in xrange(max_iterations):
            random.shuffle(train_data)
            closs = 0.0
            for seq, label in train_data:
                dy.renew_cg()
                probs = self.predict_labels(seq)

                loss = self.do_loss(probs,label)
                closs += loss.value()
                loss.backward()
                trainer.update(learning_rate)

            # check performance on dev set
            success_count = 0
            dev_closs = 0.0
            dev_size = 0
            for seq, label in dev_data:
                real_label = self.indexed_labels[label]
                prediction, dev_loss = self.classify(seq, real_label)

                # accumulate loss
                dev_closs += dev_loss

                # Avoid measuring accuracy on all good prediction of the tag
                # 'O' on NER task. This is done because most of the words has
                # this tag and therefore the accuracy might be biased. 
                if label == 'O' and (prediction == real_label):
                    continue

                success_count += (prediction == real_label)
                dev_size += 1

            avg_dev_loss = dev_closs/len(dev_data)
            
            # save accurcy and loss for current iter
            self.dev_accuracies.append(float(success_count)/dev_size)
            self.dev_losses.append(avg_dev_loss)

            # update best dev loss so far
            if avg_dev_loss < best_dev_loss:
                best_dev_loss = avg_dev_loss
                best_iter = ITER

            print "Train avg loss: %s | Dev accuracy: %s | Dev avg loss: %s" % (closs/len(train_data), float(success_count)/dev_size,
            avg_dev_loss)

            # Early stopping -
            # If the loss on dev-test has not decreased for 3 consecutive iterations - finish here
            if ITER > best_iter+2:
                break

        print 'Learning process is finished!'



if __name__ == '__main__':

    if len(sys.argv) != 2:
        raise ValueError("Exacly one argument should be supplied - task type (pos,ner)")

    task = sys.argv[1]
    if task not in ('pos', 'ner'):
        raise ValueError("Unknown task - the legal values are 'pos' or 'ner'")

    # get train&dev sets
    if task == 'pos':
        train_set = get_examples_set(POS_TRAIN_FILE)
        dev_set = get_examples_set(POS_DEV_FILE)
        learning_rate = 0.01
    # NER
    else:
        train_set = get_examples_set(NER_TRAIN_FILE)
        dev_set = get_examples_set(NER_DEV_FILE)
        learning_rate = 0.01

    # get the vocabulary and the external word embedding
    indexed_vocab, word_vectors = get_words_vec('vocab.txt', 'wordVectors.txt')

    # get labels set for the current task and index them
    labels = set([ex[1] for ex in train_set])
    indexed_labels = {l: i for i,l in enumerate(labels)}

    # For each train-word that does not exist in the vocabulary, replace it
    # with the most similar word in the vocabulary
    init_mapping = {}
    for i in range(len(train_set)):
        word, tag = train_set[i]
        if word not in indexed_vocab:
            lower_w = word.lower()
            # check if lower-case version is in the vocabulary
            if lower_w in indexed_vocab:             
                init_mapping[word] = indexed_vocab[lower_w] # save for init later

            # append the new word to the vocabulary
            indexed_vocab[word] = len(indexed_vocab)


    # add 'special words' to vocab
    word_not_exists = 'WORD_NOT_EXISTS!!'
    sequence_start = ['word_start_1', 'word_start_2'] # padd words
    sequence_end = ['word_end_1', 'word_end_2']       # padd words
    special_words = [word_not_exists]+sequence_start+sequence_end
    for word in special_words:
        indexed_vocab[word] = len(indexed_vocab)

    # prepare examples - for train set and validation set
    # assemble sequences of 5 words for each example
    # the tag of each sequence will be the tag of the word that placed in the middle
    # padd boundaries with 2 special words for each side
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    words_seq = sequence_start+[ex[0] for ex in train_set]+sequence_end
    train_data = []
    for i in range(len(train_set)):
        ex = words_seq[i:i+5], train_set[i][1]
        train_data.append(ex)

    print 'Train set preprocessing is finished'
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # handle dev-set words that are not in the vocabulary by replacing
    # them with a special word that not exists in train and dev)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    for i in range(len(dev_set)):
        word, tag = dev_set[i]
        if word not in indexed_vocab:
            dev_set[i] = word_not_exists, tag

    # prepare dev data - same as above
    words_seq = sequence_start+[ex[0] for ex in dev_set]+sequence_end
    dev_data = []
    for i in range(len(dev_set)):
        ex = (words_seq[i:i+5], dev_set[i][1])
        dev_data.append(ex)

    print 'Dev set preprocessing is finished'
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # build a new dynet model 
    my_model = dynet_model(indexed_vocab, indexed_labels, init_mapping, word_vectors)

    # train the model
    my_model.train_model(train_data, dev_data, learning_rate)

    # write dev accuracy and loss from all the iterations in a file
    my_model.write_dev_accuracies_to_file()


    # build test-set sequences
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    test_file = 'data/%s/test' % task
    print 'Preprocessing test file: %s' % test_file

    # get test words sequence from file
    with open(test_file, 'r') as f:
        data_lines = f.readlines()
    test_words = [line.strip() for line in data_lines if line != '\n']

    # save for later use
    original_test_words = test_words[:]
    
    # handle test-set words that are not in the vocabulary by replacing
    # them with a special word
    for i in range(len(test_words)):
        if test_words[i] not in indexed_vocab:
            test_words[i] = word_not_exists

    # pad test words
    test_words = sequence_start + test_words + sequence_end

    # build groups of 5 consecutive words for each test word
    test_data = []
    for i in range(2,len(test_words)-2):
        test_data.append(test_words[i-2:i+3])

    print 'Test set preprocessing is finished'
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # get predictions from the model
    preds = my_model.predict_blind_test(test_data)

    # write result file
    test_res_file = 'data/{0}/test3.{0}'.format(task)
    print 'Writing test predictions to %s' % test_res_file
    with open(test_res_file, 'w') as f:
        for w, pred in zip(original_test_words, preds):
            f.write("%s %s\n" % (w, pred))









