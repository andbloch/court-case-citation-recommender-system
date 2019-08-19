import torch
import torch.nn as nn
from torch.autograd import Variable


# SENTENCE EMBEDDER ############################################################


class SentenceEmbedder(nn.Module):

    def __init__(self,
                 word_embedding_dim,
                 num_layers,
                 sentence_embedding_dim):

        # initialize module superclass
        super(SentenceEmbedder, self).__init__()

        # keep track of supplied parameters
        self.word_embedding_dim = word_embedding_dim
        self.num_layers = num_layers
        self.sentence_embedding_dim = sentence_embedding_dim

        # define word embedding
        weight_file = open('../pretrained_word_embedding/glove.6B/weight_matrix.pt', 'rb')
        weight_matrix = torch.load(weight_file)
        num_embeddings, embedding_dim = weight_matrix.size()
        self.word_embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.word_embedding.load_state_dict({'weight': weight_matrix})
        self.word_embedding.weight.requires_grad = False

        # create GRU RNN
        # here the batch dim is used for sentences in one document
        self.gru = nn.GRU(input_size=self.word_embedding_dim,
                          hidden_size=self.sentence_embedding_dim,
                          num_layers=self.num_layers,
                          bias=True,
                          batch_first=True,
                          #dropout=0, TODO: think about dropout
                          bidirectional=False)

    def init_hidden(self, num_sentences):
        # samples initial weights of the form:
        # (sentence_embedder_num_layers, num_sentences, sentence_embedding_dim)
        dims = (self.num_layers, num_sentences, self.sentence_embedding_dim)
        init = Variable(torch.zeros(dims, device=self.gru.bias_hh_l0.device))
        return init

    def forward(self, D, sentence_lengths):
        """
        Forward propagation of network
        :param D: represents a document (as a matrix) of dimension
        (num_sentences, max_sentence_length)
        where max_sentence_length is the length of the longest sentence in D.
         - each row in D represents a sentence (=sequence of word indices)
         - each sentence is a sequence of word indices
         - the lengths of the sentences (number of word indices) is stored
           in the list sentence_lengths
        :param sentence_lengths: list of lengths of the original sentences.
        these need to be provided as D is padded for the shorter sequences
        :return: the sentence embeddings of the sentences in the document D.
        it's a tensor of the dimensions
        (num_sentences, sentence_embedding_dim)
        """

        # determine original number of sentences
        num_sentences = D.shape[0]

        # ---------------------------------
        # 1. word embedding
        # transform word indices into embedding vectors
        # TODO: shall we do this somewhere in the data-loader
        # TODO: as it's not trained anyways??
        # dim transformation:
        # (num_sentences, seq_len) ->
        # (num_sentences, seq_len, sentence_embedding_dim)
        D = self.word_embedding(D)

        # ---------------------------------
        # 2. apply RNN  (packing + forward)
        # pack padded sequence to hide padded items from GRU RNN
        # returned D will not be a tensor, but some special PackedSequence obj
        # enforce_sorted=False avoids the requirement that the sequences
        # in the batch are sorted by their length in decreasing order
        D = torch.nn.utils.rnn.pack_padded_sequence(D,
                                                    sentence_lengths,
                                                    batch_first=True,
                                                    enforce_sorted=False)
        # initialize GRU RNN hidden state
        hidden = self.init_hidden(num_sentences)
        # run packed sequence through GRU RNN
        # gives the following objects:
        # - _ (or H): a series of hidden states (still a PackedSequence object)
        # - hidden: last hidden state of each sequence (until true sequence
        #           length) of shape:
        #           (num_layers, num_sentences, sentence_embedding_dim)
        _, hidden = self.gru(D, hidden)

        # return last hidden layer outputs (of each sequence) as sentence
        # embeddings. the tensor has the shape
        # (num_sentences, sentence_embedding_dim)
        Se = hidden[self.num_layers-1,:,:]

        return Se
