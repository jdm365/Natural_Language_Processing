import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from transformer import Transformer

class Preprocesser:
    def __init__(self, filename, sentence_length, batch_size, mask_frequency, vocab_filename='words_alpha.txt'):
        self.filename = filename
        self.text = self.stringify_txt_file()
        self.vocab_filename = vocab_filename
        self.vocab, self.vocab_size = read_vocab_file()
        self.word_map = self.create_word_map()
        self.add_sep_tokens()
        self.word_list = self.text.split(' ')
        self.sentence_length = sentence_length
        self.batch_size = batch_size
        self.mask_frequency = mask_frequency


    def stringify_txt_file(self):
        with open(self.filename, 'r') as file:
            data = file.read()
            data = data.lower()
            return data

    def read_vocab_file(self): 
        with open(self.vocab_filename, 'r') as file:
            data = file.read()
            data = data.lower()
            vocab = data.splitlines()
            return vocab, len(vocab)

    def add_sep_tokens(self):
            self.text = self.text.replace('.', ' <sep>')
            self.vocab.append('<sep>')
            self.vocab_size += 1
            self.word_map['<sep>'] = self.vocab_size

    def create_word_map(self):
        self.word_map = {}
        for idx, word in self.vocab:
            self.word_map[word] = idx
        
    def mask_inputs(self, sentences):
        new_sentences = []
        tot_idxs = []
        for sentence in sentences:
            new = sentence
            n_masked = int(len(sentence) * self.mask_frequency)
            masker = np.random.choice(3, n_masked, p=[0.8, 0.1, 0.1])
            idxs = np.random.randint(1, len(new), n_masked)
            for mask, idx in zip(masker, idxs):
                if '<sep>'.isin(new[idx]):
                    continue
                elif mask == 0:
                    new[idx] = '<mask>'
                elif mask == 1:
                    new[idx] = np.random.choice(self.vocab)
                else:
                    continue
            new_sentences.append(new)
            tot_idxs.append(idxs)
        return new_sentences, sentences, tot_idxs


    def load_batch_of_strings(self):
        ## return dims (batch_size, list(sentence_length))
        start_idxs = np.random.randint(0, self.vocab_size - self.sentence_length, \
                                      self.batch_size)
        sentence_lists = ['<cls>' + self.word_list[idx:idx+self.batch_size] for idx in start_idxs]
        masked, unmasked, label_idxs  = self.mask_inputs(sentences)
        ## TODO: Make word map indexer for list and list of lists.
        masked_tokenized = self.word_map[masked]
        unmasked_tokenized = self.tokenizer.encode_as_ids(unmasked)
        return masked_tokenized, unmasked_tokenized


if __name__ == '__main__':
    preprocessor = Preprocesser(filename='datasetSentences.txt', sentence_length=512,
                                vocab_size=5000, batch_size=64, mask_frequency=0.15)

