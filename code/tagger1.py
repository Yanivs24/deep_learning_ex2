import numpy as np
import dynet as dy
import random

STUDENT={'name': 'Yaniv Sheena',
         'ID': '308446764'}


# NER task files paths
NER_TRAIN_FILE = 'data/ner/train'
NER_DEV_FILE   = 'data/ner/dev'
# POS task files paths
POS_TRAIN_FILE = 'data/pos/train'
POS_DEV_FILE   = 'data/pos/dev'


# def get_words_vec(vocab_file, wordsvec_file):
#   with open(vocab_file, 'r') as f:
#       data_lines = f.readlines()

#   # get words from the vocabulary
#   words = [w.strip() for w in data_lines]
#   w2ind = {w: i for i,w in enumerate(words)}

#   # get all pre-trained words vector (E)
#   words_vec = np.loadtxt(wordsvec_file)

#   return w2ind, words_vec

def get_examples_set(examples_file):
    examples = []
    # the order is important here - we get a sequence
    for line in open(examples_file, 'r'):
        ex = tuple(line.strip().split())
        if len(ex) == 2:
            examples.append(ex)

    return examples


class dynet_model:
    def __init__(self, indexed_vocab, indexed_labels, hid_dim=100, emb_dim=50):

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

        # some inits
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

    # Concatenating word vectors within w_sequence
    def encode_seq(self, w_sequence):
        doc = [self.indexed_vocab[w] for w in w_sequence]
        embs = [self.E[idx] for idx in doc]
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



    def write_dev_accuracies_to_file(self, path):
        print 'Writing dev results..'
        np.savetxt("%s/%s" % (path, 'dev_accuracies.txt'), np.array(self.dev_accuracies))
        np.savetxt("%s/%s" % (path, 'dev_loss.txt'), np.array(self.dev_losses))


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
                real_label = indexed_labels[label]
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

    # choose task from: 'pos' | 'ner'
    task = 'pos'

    # get train&dev sets
    if task == 'pos':
        train_set = get_examples_set(POS_TRAIN_FILE)
        dev_set = get_examples_set(POS_DEV_FILE)
        learning_rate = 0.001
    else:
        train_set = get_examples_set(NER_TRAIN_FILE)
        dev_set = get_examples_set(NER_DEV_FILE)
        learning_rate = 0.01

    # words vocabulary and labels set
    vocab = set([ex[0] for ex in train_set])
    labels = set([ex[1] for ex in train_set])

    # index words and labels using a dict
    indexed_vocab = {w: i for i,w in enumerate(vocab)}
    indexed_labels = {l: i for i,l in enumerate(labels)}

    # build clusters of all tags - for efficient word swapping in dev-test later
    words_by_tag = {}
    for word, tag in train_set:
        words_by_tag.setdefault(tag, []).append(word)

    # prepare examples - for train set and validation set
    # assemble sequences of 5 words for each example
    # the tag of each sequence will be the tag of the word that placed in the middle
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    words_seq = [ex[0] for ex in train_set]
    train_data = []
    for i in range(2, len(train_set)-2):
        ex = words_seq[i-2:i+3], train_set[i][1]
        train_data.append(ex)

    print 'Train set preprocessing is finished'
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # handle dev-set words that are not in the vocabulary by replacing
    # them with existing words that got the same tag in test-set (pick them randomly)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    for i in range(len(dev_set)):
        word, tag = dev_set[i]
        if word not in vocab:
            dev_set[i] = random.choice(words_by_tag[tag]), tag

    # prepare dev data - same as above
    words_seq = [ex[0] for ex in dev_set]
    dev_data = []
    for i in range(2, len(dev_set)-2):
        ex = (words_seq[i-2:i+3], dev_set[i][1])
        dev_data.append(ex)

    print 'Dev set preprocessing is finished'
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # build a new dynet model and train it
    my_model = dynet_model(indexed_vocab, indexed_labels)

    my_model.train_model(train_data, dev_data, learning_rate)

    my_model.write_dev_accuracies_to_file('results')


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
    # them with randomly picked existing words
    for i in range(len(test_words)):
        if test_words[i] not in vocab:
            # print 'The word %s from dev-set does not exist in the vocabulary - replace it' % word
            test_words[i] = random.choice(train_set)[0]

    # pad test words with random words from the train set (2 at each side)
    pad_words = [random.choice(train_set)[0] for _ in range(4)]
    test_words = pad_words[:2] + test_words + pad_words[2:]

    # build groups of 5 consecutive words for each test word
    test_data = []
    for i in range(2,len(test_words)-2):
        test_data.append(test_words[i-2:i+3])

    print 'Test set preprocessing is finished'
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # get predictions from the model
    preds = my_model.predict_blind_test(test_data)

    # write result file
    test_res_file = 'data/{0}/test1.{0}'.format(task)
    print 'Writing test predictions to %s' % test_res_file
    with open(test_res_file, 'w') as f:
        for w, pred in zip(original_test_words, preds):
            f.write("%s %s\n" % (w, pred))









