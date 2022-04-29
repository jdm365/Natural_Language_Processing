import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from transformer import Transformer
import sentencepiece as spm

class Preprocesser:
    def __init__(self, filename, sentence_length, vocab_size, batch_size):
        self.filename = filename
        self.vocab_size = vocab_size
        self.word_list = self.generate_word_list()
        self.tokenizer = self.create_tokenizer()
        self.batch_size = batch_size


    def generate_word_list(self):
        with open(self.filename, 'r') as file:
            data = file.read()
            data = data.lower()
            return data.split()


    def create_tokenizer(self):
        spm.SentencePieceTrainer.train(input=self.filename, model_prefix='m',\ 
                                        user_defined_symbols=['<sep>','<cls>'],\
                                        vocab_size=self.vocab_size)
        sp = spm.SentencePieceProcessor()
        sp.load('m.model')
        return sp

    def add_cls_sep_tokens(self):
        return

    
    def load_batch_of_strings(self):
        ## return dims (batch_size, list(sentence_length))
        start_idxs = np.random.randint(0, len(self.word_list) - self.sentence_length, \
                                      self.batch_size)
        sentence_lists = [self.word_list[idx:idx+self.batch_size] for idx in start_idxs]
        sentences = ' '.join([str(item) for item in sentence_lists])
        tokenized = self.tokenizer.encode_as_ids(sentences)
        return tokenized



