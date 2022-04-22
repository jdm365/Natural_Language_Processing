from curses import window
import numpy as np
from sympy import julia_code
from tqdm import tqdm
from itertools import chain
import random
from utils import sigmoid, softmax

class Dataset:
    def __init__(self, filename, header=True):
        self.filename = filename
        self.header = int(header)
        self.dataset = self.read_file_to_dataset()

        self.vocab, self.vocab_size = self.get_vocab()
        self.word_map = self.get_mappings()
        self.word_counts, self.length = self.get_word_counts()
        self.augmented_data = self.augment_data()
        self.weighted_samples = self.weight_words()

    def read_file_to_dataset(self):
        print('... reading into dataset ...')
        sentences = []
        reviews = open(self.filename).readlines()
        n_reviews = len(reviews)

        for line in range(self.header, n_reviews):
            sentence = [word.lower() for word in reviews[line].split()[1:]]
            sentences.append(sentence)
        return sentences

    def get_vocab(self):
        print('... identifying vocabulary ...')
        all_words = []
        for sentence in self.dataset:
            for word in sentence:
                all_words.append(word)
        vocab = list(set(all_words))
        return vocab, len(vocab)

    def get_mappings(self):
        mapping = {}
        for idx, word in enumerate(self.vocab):
            mapping[word] = idx
        return mapping

    def get_word_counts(self):
        print('... getting word counts ...')
        counts = {}
        words = list(chain(*self.dataset))
        for word in tqdm(self.vocab):
            counts[word] = words.count(word)
        return counts, len(words)

    def augment_data(self, threshold=1e-5, N=30, min_length=3):
        print('... augmenting data ...')
        p_reject = np.zeros(self.vocab_size, dtype=np.float32)
        for idx, word in enumerate(self.vocab):
            freq = self.word_counts[word] / self.length
            p_reject[idx] = max(0, 1 - np.sqrt(threshold / freq))
        all_sentences = []
        for sentence in tqdm(self.dataset*N):
            new_sentence = []
            for word in sentence:
                if p_reject[self.word_map[word]] == 0 or \
                    random.random() >= p_reject[self.word_map[word]]:
                    new_sentence.append(word)
            if len(new_sentence) >= min_length:
                all_sentences.append(new_sentence)
        return all_sentences

    def get_random_context(self, length=5):
        sentence = self.augmented_data[np.random.randint(0, len(self.augmented_data))]
        context_len = np.random.randint(1, max(2, min(length, (len(sentence)-1)//2)))
        cw_index = np.random.randint(context_len, len(sentence)-context_len)
        center_word = sentence[cw_index]
        context_low = sentence[cw_index-context_len:cw_index]
        context_high = sentence[cw_index+1:cw_index+context_len+1]
        context = context_low + context_high
        return center_word, context

    def weight_words(self):
        probs = np.zeros_like(self.vocab, dtype=np.float32)
        for idx, word in enumerate(self.vocab):
            probs[idx] = (self.word_counts[word] / self.length) ** 3/4
        return probs

    def sample_word_idx(self):
        word = np.random.choice(self.vocab, p=self.weighted_samples)
        idx = self.word_map[word]
        return idx

    def get_negative_samples(self, outside_word_idx, n_samples=10):
        neg_sample_word_indices = []
        while len(neg_sample_word_indices) != n_samples:
            new_idx = self.sample_word_idx()
            if new_idx != outside_word_idx:
                neg_sample_word_indices.append(new_idx)
        return neg_sample_word_indices


class Word2Vec:
    def __init__(self, window_size, vec_size, dataset):
        self.data = dataset
        self.window_size = window_size
        self.vocab_size = self.data.vocab_size
        self.vec_size = vec_size
        self.word_map = self.data.word_map

        self.center_word_vectors = np.random.normal(size=(self.vocab_size, self.vec_size), \
                                                    loc=0.0, scale=0.01).astype(np.float32)
        self.outside_word_vectors = np.random.normal(size=(self.vocab_size, self.vec_size), \
                                                    loc=0.0, scale=0.01).astype(np.float32)
     
    def softmax_loss_and_gradient(self, center_word_vector, outside_word_index):
        theta = np.dot(self.outside_word_vectors, center_word_vector)
        p = np.zeros((self.vocab_size), dtype=np.float32)
        p[outside_word_index] = 1.0
        p_hat = softmax(theta)
        loss = -np.sum(p*np.log(p_hat))
        error = p_hat - p

        grad_outside_vectors = np.outer(error, center_word_vector)
        grad_center_vector = np.dot(error, self.outside_word_vectors)
        return loss, grad_center_vector, grad_outside_vectors

    def skipgram(self, center, outside):
        center_vector = self.center_word_vectors[self.word_map[center]]
        outside_indices = [self.word_map[word] for word in outside]

        loss = 0
        grad_center = np.zeros(self.center_word_vectors.shape, dtype=np.float32)
        grad_outside = np.zeros(self.outside_word_vectors.shape, dtype=np.float32)

        for idx in outside_indices:
            J, dJ_dWc, dJ_dWo = self.softmax_loss_and_gradient(center_vector, \
                                            outside_word_index=idx)
            loss += J
            grad_center[self.word_map[center]] += dJ_dWc
            grad_outside += dJ_dWo
        return loss, grad_center, grad_outside

    def neg_sampling_loss(self, center_word_vec, outside_word_idx, n_samples=5):
        neg_sample_idx = self.data.get_negative_samples(outside_word_idx, n_samples)
        grad_outside_vecs = np.zeros_like(self.outside_word_vectors, dtype=np.float32)

        w_c = center_word_vec
        w_o = self.outside_word_vectors[outside_word_idx]
        w_k = self.outside_word_vectors

        theta = np.dot(w_o.T, w_c)
        loss = -np.log(sigmoid(theta)) - np.sum(np.log(sigmoid(-np.dot(w_c, w_k.T))))
        grad_center_vec = -w_o * (1 - sigmoid(theta))
        grad_outside_vecs[outside_word_idx] = w_c * (1 - sigmoid(theta))
        for idx in neg_sample_idx:
            outside_word_vec = self.outside_word_vectors[idx]
            beta = np.dot(-w_c, outside_word_vec.T)
            grad_center_vec += np.dot(1 - sigmoid(beta), outside_word_vec)
            grad_outside_vecs[idx] += w_c * (1 - sigmoid(beta))
        return loss, grad_outside_vecs, grad_center_vec

    def apply_gradients(self, grad_c, grad_o, lr=1e-2):
        self.outside_word_vectors -= lr * grad_o
        self.center_word_vectors -= lr * grad_c

class SGDWrapper:
    def __init__(self, n_iters=20000, batch_size=50, anneal=10000):
        self.n_iters = n_iters
        self.batch_size = batch_size
        self.anneal = anneal

    def sgd(self, word2vec, dataset, lr, decay_rate, window):
        for iteration in range(self.n_iters):
            loss = 0.0
            grad_c = np.zeros_like((word2vec.center_word_vectors))
            grad_o = np.zeros_like((word2vec.outside_word_vectors))
            for batch in range(self.batch_size):
                center_word, context = dataset.get_random_context(dataset.augmented_data, window)
                cost, d_c, d_o = word2vec.skipgram(center_word, context)
                loss += cost / self.batch_size
                grad_c += d_c / self.batch_size
                grad_o += d_o / self.batch_size
            word2vec.apply_gradients(grad_c, grad_o, lr)
            if iteration % self.anneal == 0 and iteration != 0:
                lr *= decay_rate
            if iteration % 100 == 0:
                print(f'Iteration: {iteration+1} \t Loss: {loss}')

if __name__ == '__main__':
    dataset = Dataset(filename='datasetSentences.txt')
    word2vec = Word2Vec(window_size=5, vec_size=50, dataset=dataset)
    runner = SGDWrapper()
    runner.sgd(word2vec, dataset, lr=0.3, decay_rate=0.5, window=word2vec.window_size)