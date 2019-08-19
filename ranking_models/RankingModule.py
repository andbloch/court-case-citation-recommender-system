import torch
import torch.nn as nn

# TODO: combine desired logic of all models here
# TODO: (say: public documentation, private implementation)


# ABSTRACT RANKING MODULE ######################################################


class RankingModule(nn.Module):

    def __init__(self):

        # initialize 'Module' superclass
        super(RankingModule, self).__init__()


# ABSTRACT RANKING FUNCTIONS ###################################################


    def rank_citation_pairs(self, citing, cited):
        """
        Computes the ranks of citing-cited pairs.

        user_idxs      item_idxs                result
        ---------      ---------           ---------------_
         ([u_1]         [i_1])             [rank(u_1, i_1)]
         ([u_2]         [i_2])    |--->    [rank(u_2, i_2)]
            .     ,       .
         ([u_n]         [i_n])             [rank(u_n, i_n)]

        :param user_idxs: the user_idxs of the pairs
        :param item_idxs: the item_idxs of the pairs
        :return: a tensor containing the ranks for each pair
        """
        raise NotImplementedError


# ABSTRACT RUN NAME HELPERS ####################################################


    def get_net_name(self):
        """
        Returns the name of the network
        :return: name of the network
        """
        raise NotImplementedError


    def get_run_name(self):
        """
        Returns the run name based on the model's configuration
        :return: run name string
        """
        raise NotImplementedError


# FORWARD PROPAGATION VARIANTS #################################################


    def forward(self, citing, cited):
        """
        Does the forward propagation of the NN.
        It selects a ranking function based on the values and dimensions of the
        two arguments (see logic in code). This somewhat overloads the forward
        function but the advantage of this is that we can exploit the
        DataParallel module (do several forms of predictions on multiple GPUs).
        :param citing: user_idxs to predict ranks for
        :param cited: item_idxs to predict ranks for
        :return: the ranks for each pair
        """

        ranks = None
        # if we have no items provided predict for the supplied user_ids
        # and all items
        if citing[0].shape[0] == 1 and cited is None:
            ranks = self.rank_all_cases(citing)
        # if we have both provided predict user-item pairs or sets
        else:
            ranks = self.predict_for_citing_and_subsets_of_items(citing,
                                                                 cited)
        return ranks


    def predict_for_citing_and_subsets_of_items(self, citing, cited_sets):
        """
        Predicts the rankings of for users and their corresponding item sets.
        So we have the mapping

                user_idxs             item_idx_sets
                ---------    --------------------------------
                  [u_1]      [i_(1,1), i_(1,2), ..., i_(1,m)]
                    .     ,                      .
                  [u_n]      [i_(n,1), i_(n,2), ..., i_(n,m)]

                                   |--->

                                   result
        ------------------------------------------------------------------
        [rank(u_1, i_(1,1)), rank(u_1, i_(1,2)), ... , rank(u_1, i_(1,m))]
                                                  .
        [rank(u_1, i_(n,1)), rank(u_1, i_(n,2)), ... , rank(u_n, i_(1,m))]

        where u_1,...,u_n are a subset of all users and i_(j,1),...,i_(j,m)
        the subset of all items of the j-th user u_j to predict the ranks for.

        Note that user_idxs and item_idxs may just contain one user or item.
        The cases covered are
        - (1) one user, one item            (one pair)
        - (2) one user, many items          (several implicit pairs)
        - (3) many users, one item          (several implicit pairs)
        - (4) equally many users and items  (pairs)
        - (5) many users, with item sets    (several implicit pairs)

        :param citing: the user idxs to predict the rankings for
        :param item_idxs: the item idx sets corresponding to the user to
        predict the rankings for.
        :return: the rankings
        """

        # create array of ranks
        # (batch_size, num_cited_documents)
        dims = (citing[0].shape[0], len(cited_sets))
        ranks = torch.empty(dims, device=citing[0].device)

        num_cited = len(cited_sets)
        # for the i-th item in each item set
        for i in range(num_cited):
            # compute ranks
            ranks[:, i] =  self.rank_citation_pairs(citing, cited_sets[i])

        return ranks