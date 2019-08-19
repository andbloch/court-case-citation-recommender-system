import torch
import torch.nn as nn
from torch.autograd import Variable


# DOCUMENT EMBEDDER ############################################################


class DocumentEmbedder(nn.Module):

    def __init__(self,
                 sentence_embedding_dim,
                 num_layers,
                 document_embedding_dim):

        # initialize module superclass
        super(DocumentEmbedder, self).__init__()

        # keep track of supplied parameters
        self.sentence_embedding_dim = sentence_embedding_dim
        self.num_layers = num_layers
        self.document_embedding_dim = document_embedding_dim

        # create GRU RNN
        self.gru = nn.GRU(input_size=self.sentence_embedding_dim,
                          hidden_size=self.document_embedding_dim,
                          num_layers=self.num_layers,
                          bias=True,
                          batch_first=True,
                          #dropout=0, TODO: think about dropout
                          bidirectional=False)

    def init_hidden(self, num_documents):
        # samples initial weights of the form:
        # (sentence_embedder_num_layers, num_documents, sentence_embedding_dim)
        dims = (self.num_layers, num_documents, self.document_embedding_dim)
        init = Variable(torch.zeros(dims, device=self.gru.bias_hh_l0.device))

        return init

    def forward(self, Se, document_lengths):
        """
        Forward propagation function of network
        :param Se: represents a batch of documents represented as a sequence of
        sentence embeddings. Se has the dimensions
        (num_documents, max_document_length)
        where max_document_length is equal to the maximum number of sentences
        in a document.
         - each row in Se represents a document (=seq. of sentence embeddings)
         - each document is a sequence of sentence embeddings
         - the lengths of the documents (number of sentence embeddings) is
           stored in the list document_lengths
        :param document_lengths: list of lengths of the original documents.
        these need to be provided as Se is padded for the shorter sequences
        :return: the document embeddings of the documents in Se.
        it's a tensor of the dimensions
        (num_documents, document_embedding_dim)
        :return:
        """

        # determine number of documents
        num_documents = Se.shape[0]

        # pack padded sequence to hide padded items from GRU RNN
        # returned D will not be a tensor, but some special PackedSequence obj
        # enforce_sorted=False avoids the requirement that the sequences
        # in the batch are sorted by their length in decreasing order
        Se = torch.nn.utils.rnn.pack_padded_sequence(Se,
                                                     document_lengths,
                                                     batch_first=True,
                                                     enforce_sorted=False)

        # initialize GRU RNN hidden state
        hidden = self.init_hidden(num_documents)

        # run packed sequence through GRU RNN
        # gives the following objects:
        # - _     : a series of hidden states (still a PackedSequence object)
        # - hidden: last hidden state of each sequence (until true sequence
        #           length) of document. hidden has the shape:
        #           (num_layers, num_documents, document_embedding_dim)
        _, hidden = self.gru(Se, hidden)

        # return last hidden layer outputs (of each sequence) as document
        # embeddings. the tensor has the shape
        # (num_documents, document_embedding_dim)
        De = hidden[self.num_layers-1,:,:]

        return De
