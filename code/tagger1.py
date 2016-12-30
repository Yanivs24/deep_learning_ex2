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
# 	with open(vocab_file, 'r') as f:
# 		data_lines = f.readlines()

# 	# get words from the vocabulary
# 	words = [w.strip() for w in data_lines]
# 	w2ind = {w: i for i,w in enumerate(words)}

# 	# get all pre-trained words vector (E)
# 	words_vec = np.loadtxt(wordsvec_file)

# 	return w2ind, words_vec

def get_examples_set(examples_file):
	examples = []
	# the order is important here - we get a sequence
	for line in open(examples_file, 'r'):
		ex = tuple(line.strip().split(" "))
		if len(ex) == 2:
			examples.append(ex)

	return examples

def encode_words(indexed_vocab, E, sentence):
	 doc = [indexed_vocab[w] for w in sentence]
	 embs = [E[idx] for idx in doc]
	 return dy.concatenate(embs) 

def train_model(indexed_vocab, indexed_labels, train_data, dev_data, hid_dim_size=100, emb_dim=50):

	out_dim = len(indexed_labels)
	vocab_size = len(indexed_vocab)	

	# define the parameters
	model = dy.Model()

	# first layer params
	pW1 = model.add_parameters((hid_dim_size, 5*emb_dim))
	pb1 = model.add_parameters(hid_dim_size)

	# hidden layer params
	pW2 = model.add_parameters((out_dim, hid_dim_size))
	pb2 = model.add_parameters(out_dim)

	# word embedding - E
	E = model.add_lookup_parameters((vocab_size,emb_dim))


	def predict_labels(w_sequence):
		x = encode_seq(w_sequence)
		h = layer1(x)
		y = layer2(h)
		return dy.softmax(y)

	def layer1(x):
		W = dy.parameter(pW1)
		b = dy.parameter(pb1)
		return dy.tanh(W*x+b)

	def layer2(x):
		W = dy.parameter(pW2)
		b = dy.parameter(pb2)
		return W*x+b

	# Concatenating word vectors
	def encode_seq(w_sequence):
		doc = [indexed_vocab[w] for w in w_sequence]
		embs = [E[idx] for idx in doc]
		return dy.concatenate(embs) 

	def do_loss(probs, label):
 		label = indexed_labels[label]
 		return -dy.log(dy.pick(probs,label))

 	def classify(w_sequence):
		dy.renew_cg()
		probs = predict_labels(w_sequence)
		vals = probs.npvalue()
		return np.argmax(vals)

 	trainer = dy.SimpleSGDTrainer(model)
	closs = 0.0
	for ITER in xrange(1000):
		random.shuffle(train_data)
	 	for seq, label in train_data:
	 		dy.renew_cg()
	 		probs = predict_labels(seq)

	 		loss = do_loss(probs,label)
	 		closs += loss.value()
	 		loss.backward()
	 		trainer.update()

		success_count = 0
		for seq, label in dev_data:
			success_count += classify(seq) == indexed_labels[label]

		print "Train avg loss: %s | Validation rate: %s" % (closs/len(train_data), float(success_count)/len(dev_data))

if __name__ == '__main__':

	# get train&dev sets
	train_set = get_examples_set(POS_TRAIN_FILE)
	dev_set = get_examples_set(POS_DEV_FILE)

	vocab = set([ex[0] for ex in train_set])
	labels = set([ex[1] for ex in train_set])

	# index words and labels using a dict
	indexed_vocab = {w: i for i,w in enumerate(vocab)}
	indexed_labels = {l: i for i,l in enumerate(labels)}

	# prepare examples - for train set and validation set
	# assemble in sequences of 5 words for each example
	# the tag wof each sequence will be the tag of the word in the middle
	words_seq = [ex[0] for ex in train_set]
	train_data = []
	for i in range(2, len(train_set)-2):
		ex = (words_seq[i-2:i+3], train_set[i][1])
		train_data.append(ex)

	words_seq = [ex[0] for ex in dev_set]
	dev_data = []
	for i in range(2, len(dev_set)-2):
		ex = (words_seq[i-2:i+3], dev_set[i][1])
		dev_data.append(ex)


	train_model(indexed_vocab, indexed_labels, train_data, dev_data)








