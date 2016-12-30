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

def pick_random_word_with_tag(words_by_tag, tag):
	''' Used to avoid using words that do not exists in the vocab'''
	random.choice(words_by_tag[tag])


def train_model(indexed_vocab, indexed_labels, train_data, dev_data, hid_dim=100, emb_dim=50):

	vocab_size = len(indexed_vocab)
	out_dim = len(indexed_labels)

	# define the parameters
	model = dy.Model()

	# first layer params
	pW1 = model.add_parameters((hid_dim, 5*emb_dim))
	pb1 = model.add_parameters(hid_dim)

	# hidden layer params
	pW2 = model.add_parameters((out_dim, hid_dim))
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

 	def classify(w_sequence, label):
		dy.renew_cg()
		probs = predict_labels(w_sequence)
		vals = probs.npvalue()
		return np.argmax(vals), -np.log(vals[label])

	# train a model
 	trainer = dy.SimpleSGDTrainer(model)
 	best_dev_loss = 1e3
 	best_iter = 0
	for ITER in xrange(100):
		random.shuffle(train_data)
		closs = 0.0
	 	for seq, label in train_data:
	 		dy.renew_cg()
	 		probs = predict_labels(seq)

	 		loss = do_loss(probs,label)
	 		closs += loss.value()
	 		loss.backward()
	 		trainer.update(0.001)

	 	# check performance on dev set
		success_count = 0
		dev_closs = 0.0
		for seq, label in dev_data:
			real_label = indexed_labels[label]
			prediction, dev_loss = classify(seq, real_label)
			success_count += (prediction == real_label)
			dev_closs += dev_loss

		avg_dev_loss = dev_closs/len(dev_data)
		
		# update best dev loss so far
		if avg_dev_loss < best_dev_loss:
			best_dev_loss = avg_dev_loss
			best_iter == ITER

		print "Train avg loss: %s | Dev accuracy: %s | Dev avg loss: %s" % (closs/len(train_data), float(success_count)/len(dev_data),
		avg_dev_loss)

		# Early stopping
		# If the loss on dev-test has not decreased for 3 consecutive iterations - finish here
		if ITER > best_iter+2:
			break


if __name__ == '__main__':

	# get train&dev sets
	train_set = get_examples_set(POS_TRAIN_FILE)
	dev_set = get_examples_set(POS_DEV_FILE)

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
	words_seq = [ex[0] for ex in train_set]
	train_data = []
	for i in range(2, len(train_set)-2):
		ex = words_seq[i-2:i+3], train_set[i][1]
		train_data.append(ex)

	print 'Train set preprocessing is finished'

	# handle dev-set words that are not in the vocabulary by replacing
	# them with existing words that got the same tag in test-set (pick them randomly)
	for i in range(len(dev_set)):
		word, tag = dev_set[i]
		if word not in vocab:
			print 'The word %s from dev-test does not exist in the vocabulary - replace it' % word
			dev_set[i] = random.choice(words_by_tag[tag]), tag

	# prepare dev data - same as above
	words_seq = [ex[0] for ex in dev_set]
	dev_data = []
	for i in range(2, len(dev_set)-2):
		ex = (words_seq[i-2:i+3], dev_set[i][1])
		dev_data.append(ex)

	print 'Data preprocessing is finished - start training the model'

	train_model(indexed_vocab, indexed_labels, train_data, dev_data)








