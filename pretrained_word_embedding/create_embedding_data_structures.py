"""

This code takes the downloaded glove.6B.200d.txt word embedding and creates
python datastructures for:

 - word2idx: maps a word to its index

 - idx2vec: maps an index to its embedding vector

 Thanks to:
 https://medium.com/@martinpella/how-to-use-pre-trained-word-embeddings-in-pytorch-71ca59249f76

"""

import numpy as np
import pickle
import bcolz
from tqdm import tqdm
import torch

# TODO: read this from configuration
WORD_EMBEDDING_DIM = 50

# create vocabulary (list of words)
vocabulary = []
# create word to index mapping
word2idx = {}
# crate index to embedding vector mapping
idx2vec = bcolz.carray(np.zeros(1), rootdir='idx2vec.dat', mode='w')

# define initial word index
# (index zero is the '<PAD>' word)
idx = 0
vocabulary.append('<PAD>')
word2idx['<PAD>'] = idx
# TODO: determine what the <pad> vector should be like!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# TODO: is that right?
idx2vec.append(np.zeros(WORD_EMBEDDING_DIM).astype(np.float))
idx = 1
GLOVE_FILE = 'glove.6B/glove.6B.'+str(WORD_EMBEDDING_DIM)+'d.txt'
with open(GLOVE_FILE, 'rb') as f:
    # determine number of lines (should be 400K TODO: + 1 pad word?)
    num_lines = sum(1 for line in open(GLOVE_FILE, 'rb'))
    # for each line in the file
    for l in tqdm(f, total=num_lines):
        # split the word and coefficients into a list
        line = l.decode().split()
        # determine the actual word (in text)
        word = line[0]
        # append word to list of vocabulary
        vocabulary.append(word)
        # create mapping of word to index
        word2idx[word] = idx
        # determine embedding vector of word
        vect = np.array(line[1:]).astype(np.float)
        # create a mapping of index to embedding vector
        idx2vec.append(vect)
        # go to next word/index
        idx += 1

# save the vocabulary
pickle.dump(vocabulary, open('glove.6B/vocabulary.pkl', 'wb'))
# save the word to index mapping
pickle.dump(word2idx, open('glove.6B/word2idx.pkl', 'wb'))
# save the index to embedding vector mapping
idx2vec = bcolz.carray(idx2vec[1:].reshape((400002, WORD_EMBEDDING_DIM)),
                       rootdir='glove.6B/idx2vec.dat', mode='w')
idx2vec.flush()

# create glove mapping
glove = {w: idx2vec[word2idx[w]] for w in vocabulary}

# TODO: we might think about reducing this to the vocabulary that is actually
# TODO: used, and also map unknown words to the <unk> vector (last one)
# TODO: but for now it's like this

# create word embedding weight matrix (for pytorch embedding layer)
weight_matrix = torch.zeros((400002, WORD_EMBEDDING_DIM))
for i, word in enumerate(vocabulary):
    weight_matrix[i] = torch.tensor(glove[word])

# save word embedding weight matrix
torch.save(weight_matrix, open('glove.6B/weight_matrix.pt', 'wb'))
