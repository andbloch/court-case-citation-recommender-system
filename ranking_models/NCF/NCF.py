from ranking_models import RankingModule
import torch

from .GMF import GMF
from .MLP import MLP


# NCF ##########################################################################


class NCF(RankingModule):


    def __init__(self,
                 num_factors,
                 num_layers,
                 word_embedding_dim,
                 gmf_sentence_embedder_num_layers,
                 gmf_sentence_embedder_dim,
                 gmf_document_embedder_num_layers,
                 mlp_sentence_embedder_num_layers,
                 mlp_sentence_embedder_dim,
                 mlp_document_embedder_num_layers):

        # initialize 'RankingModule' superclass
        super(NCF, self).__init__()

        # keep track of arguments
        self.num_layers = num_layers
        self.num_factors = num_factors

        # create GMF
        self.GMF = GMF(num_factors,
                       word_embedding_dim,
                       gmf_sentence_embedder_num_layers,
                       gmf_sentence_embedder_dim,
                       gmf_document_embedder_num_layers,
                       create_output_layer_params=False)

        # create MLP
        self.MLP = MLP(num_factors,
                       num_layers,
                       word_embedding_dim,
                       mlp_sentence_embedder_num_layers,
                       mlp_sentence_embedder_dim,
                       mlp_document_embedder_num_layers,
                       create_output_layer_params=False)

        # create parameters for output layer
        self.affine_output = torch.nn.Linear(in_features=2*num_factors,
                                             out_features=1)


# RANKING FUNCTIONS ############################################################


    def rank_citation_pairs(self, citing, cited):

        # get forward-propagation until last layer from GMF
        out_GMF = self.GMF.inner_forward(citing, cited)

        # get forward-propagation until last layer from MLP
        out_MLP = self.MLP.inner_forward(citing, cited)

        # concatenate outputs
        out = torch.cat([out_GMF, out_MLP], dim=-1)

        # apply last final linear (output) layer
        xh_ij_logits = self.affine_output(out)
        xh_ij = xh_ij_logits

        # collapse last dimension
        xh_ij = xh_ij.squeeze(1)

        return xh_ij


# RUN NAME HELPERS #############################################################


    def get_net_name(self):
        return 'NCF'


    def get_run_name(self):
        return 'L'+str(self.num_layers)+'_F'+str(self.num_factors)
