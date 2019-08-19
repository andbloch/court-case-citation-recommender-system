from ranking_models import RankingModule
import torch
from torch.nn import Linear
from document_embedding import HRNN_Embedder


# MULTI-LAYER PERCEPTRON (MLP) FOR NCF #########################################


class MLP(RankingModule):


    def __init__(self,
                 num_factors,
                 num_layers,
                 word_embedding_dim,
                 sentence_embedder_num_layers,
                 sentence_embedder_dim,
                 document_embedder_num_layers,
                 create_output_layer_params=True):

        # initialize 'RankingModule' superclass
        super(MLP, self).__init__()

        # keep track of supplied arguments
        self.num_factors = num_factors
        self.num_layers = num_layers
        self.word_embedding_dim = word_embedding_dim
        self.sentence_embedder_num_layers = sentence_embedder_num_layers
        self.sentence_embedder_dim = sentence_embedder_dim
        self.document_embedder_num_layers = document_embedder_num_layers

        # create item embedding
        self.item_embedder = HRNN_Embedder(
            word_embedding_dim=word_embedding_dim,
            sentence_embedder_num_layers=self.sentence_embedder_num_layers,
            sentence_embedding_dim=self.sentence_embedder_dim,
            document_embedder_num_layers=self.document_embedder_num_layers,
            document_embedding_dim=self.num_factors*(2**(self.num_layers-1))
        )

        # create list of parameters for fully connected layers
        # each layer should halve the size until we reach num_factors
        self.fc_layers = torch.nn.ModuleList()
        for i in range(num_layers):
            in_size = num_factors * (2**(num_layers-i))
            out_size = num_factors * (2**(num_layers-1-i))
            self.fc_layers.append(Linear(in_size, out_size))

        # create parameters for output layer
        if create_output_layer_params:
            self.affine_output = torch.nn.Linear(in_features=num_factors,
                                                 out_features=1)
            self.logistic_output = torch.nn.Sigmoid()


# UNTIL LAST LAYER FORWARD PROP ################################################


    def inner_forward(self, citing, cited):

        # embed citing and cited item
        x_i = self.item_embedder(citing[1], citing[2], citing[3])
        x_j = self.item_embedder(cited[1], cited[2], cited[3])

        # concatenate the embeddings
        x_ij = torch.cat([x_i, x_j], dim=-1)

        # propagate concatenated embedding vectors through each layer
        out = x_ij
        for fc_layer in self.fc_layers:
            out = fc_layer(out)
            out = torch.nn.ReLU()(out)
            # TODO: batch-norm and dropout

        # return input for output layer of NCF
        return out


# RANKING FUNCTIONS ############################################################


    def rank_citation_pairs(self, citing, cited):

        # get forward-propagation until last layer
        out = self.inner_forward(citing, cited)

        # apply last final linear (output) layer
        xh_ij_logits = self.affine_output(out)
        xh_ij = xh_ij_logits

        # collapse last dimension
        xh_ij = xh_ij.squeeze(1)

        return xh_ij


# RUN NAME HELPERS #############################################################


    def get_net_name(self):
        return 'MLP'


    def get_run_name(self):
        return 'L'+str(self.num_layers)+'_F'+str(self.num_factors)

