import torch
import torch.nn as nn

from .SentenceEmbedder import SentenceEmbedder
from .DocumentEmbedder import DocumentEmbedder


# HIERARCHICAL RNN EMBEDDER ####################################################


class HRNN_Embedder(nn.Module):


    def __init__(self,
                 word_embedding_dim,
                 sentence_embedder_num_layers,
                 sentence_embedding_dim,
                 document_embedder_num_layers,
                 document_embedding_dim):

        # initialize module superclass
        super(HRNN_Embedder, self).__init__()

        # keep track of supplied parameters
        self.word_embedding_dim = word_embedding_dim
        self.sentence_embedder_num_layers = sentence_embedder_num_layers
        self.sentence_embedding_dim = sentence_embedding_dim
        self.document_embedder_num_layers = document_embedder_num_layers
        self.document_embedding_dim = document_embedding_dim

        # create sentence embedder
        self.sentence_embedder = \
            SentenceEmbedder(self.word_embedding_dim,
                             self.sentence_embedder_num_layers,
                             self.sentence_embedding_dim)

        # create document embedder
        self.document_embedder = \
            DocumentEmbedder(self.sentence_embedding_dim,
                             self.document_embedder_num_layers,
                             self.document_embedding_dim)


    def forward(self, D, document_lengths, sentence_lengths):

        # D is a corpus (tensor) of documents of the shape:
        # (num_documents, max_num_sentences, max_max_num_words), where
        # - num_documents is the number of documents
        # - max_num_sentences is the maximal number of sentences in any document
        # - max_max_num_words is the max. number of words in any sentence
        # document_lengths is a list of lists [doc][num_sentences]
        # sentence_lengths is a list of lists of lists [doc][sent][num_words]

        # create tensor to hold sentence embeddings of shape:
        # (num_documents, max_num_sentences)
        num_documents, max_num_sentences, _ = D.size()
        dims = (num_documents, max_num_sentences, self.sentence_embedding_dim)
        Se = torch.zeros(dims, device = D.device)

        # TODO: can this step somehow be parallelized?
        # TODO: maybe use DATA-PARALLEL network?
        # for each document in the corpus
        for i in range(num_documents):
            # reduce view onto padded sentences
            padded_sentences = D[i,0:document_lengths[i],:]
            # determine sentence embedding for that document
            sentence_embedding = \
                self.sentence_embedder(padded_sentences,
                                       sentence_lengths[i])
            # store sentence embeddings in embedding matrix
            Se[i,0:document_lengths[i]] = sentence_embedding

        # determine document embeddings from sentence embeddings
        De = self.document_embedder(Se, document_lengths)

        return De
