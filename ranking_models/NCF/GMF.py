from ranking_models import RankingModule

import torch
from torch.nn import Linear
from torch.nn import Sigmoid

from document_embedding import HRNN_Embedder


# GENERALIZED MATRIX FACTORIZATION (GMF) FOR NCF ###############################


class GMF(RankingModule):


    def __init__(self,
                 num_factors,
                 word_embedding_dim,
                 gmf_sentence_embedder_num_layers,
                 gmf_sentence_embedder_dim,
                 gmf_document_embedder_num_layers,
                 create_output_layer_params=True):

        # initialize 'RankingModule' superclass
        super(GMF, self).__init__()

        # keep track of supplied arguments
        self.num_factors = num_factors
        self.word_embedding_dim = word_embedding_dim
        self.sentence_embedder_num_layers = gmf_sentence_embedder_num_layers
        self.sentence_embedder_dim = gmf_sentence_embedder_dim
        self.document_embedder_num_layers = gmf_document_embedder_num_layers
        self.create_output_layer_params = create_output_layer_params


        # create item embeddings
        self.item_embedder = HRNN_Embedder(
            word_embedding_dim=self.word_embedding_dim,
            sentence_embedder_num_layers=self.sentence_embedder_num_layers,
            sentence_embedding_dim=self.sentence_embedder_dim,
            document_embedder_num_layers=self.document_embedder_num_layers,
            document_embedding_dim=self.num_factors
        )

        # create parameters for output layer
        if create_output_layer_params:
            self.affine_output = Linear(in_features=num_factors,
                                        out_features=1)
            self.logistic_output = Sigmoid()


# UNTIL LAST LAYER FORWARD PROP ################################################


    def inner_forward(self, citing, cited):

        # get user and item embeddings
        x_i = self.item_embedder(citing[1], citing[2], citing[3])
        x_j = self.item_embedder(cited[1], cited[2], cited[3])

        # build element-wise product
        xh_ij_elementwise_prod = torch.mul(x_i, x_j)

        # return input for output layer
        return xh_ij_elementwise_prod


# RANKING FUNCTIONS ############################################################


    def rank_citation_pairs(self, citing, cited):

        # get forward propagation until last layer
        # (element-wise product of embeddings)
        xh_ij_elementwise_prod = self.inner_forward(citing, cited)

        # apply last final linear (output) layer
        xh_ij_logits = self.affine_output(xh_ij_elementwise_prod)
        xh_ij = self.logistic_output(xh_ij_logits)

        # collapse last dimension
        xh_ij = xh_ij.squeeze(1)

        return xh_ij


# RUN NAME HELPERS #############################################################


    def get_net_name(self):
        return 'GMF'


    def get_run_name(self):
        # TODO: add more
        return 'F'+str(self.num_factors)

