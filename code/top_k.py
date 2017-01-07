import numpy as np

STUDENT={'name': 'Yaniv Sheena',
         'ID': '308446764'}


def get_words_vec(vocab_file, wordsvec_file):
    with open(vocab_file, 'r') as f:
      data_lines = f.readlines()

    # get words from the vocabulary
    words = [w.strip() for w in data_lines]
    w2ind = {w: i for i,w in enumerate(words)}

    # get all pre-trained words vector (E)
    words_vec = np.loadtxt(wordsvec_file)

    return w2ind, words_vec


def most_similar(w2i, word_vectors, word, k):

    cos_dist = lambda v1, v2: np.abs(np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)))

    dist = []
    for w in w2i:
        # skip word
        if w == word:
            continue
        dist.append((cos_dist(word_vectors[w2i[w]], word_vectors[w2i[word]]), w))

    # take best k values (max values - cos distance)
    dist.sort(reverse=True)
    return dist[:k]



if __name__ == '__main__':
    w2i, word_vectors = get_words_vec('vocab.txt', 'wordVectors.txt')

    # example
    print most_similar(w2i, word_vectors, 'dog', 5)




