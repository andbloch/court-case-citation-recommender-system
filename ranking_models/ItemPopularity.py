from ranking_models import RankingModule
from torch.nn import Parameter


# ITEM POPULARITY MODEL ########################################################


class ItemPopularity(RankingModule):


    def __init__(self,
                 ranked_items,
                 popularities):

        # initialize 'RankingModule' superclass
        super(ItemPopularity, self).__init__()

        # create list of ranked items (sorted by their order)
        # [first item, second item, ... ]
        self.ranked_items = Parameter(ranked_items , requires_grad=False)

        # keep track of popularity of items
        # item_id -> popularity (density)
        self.popularities = Parameter(popularities , requires_grad=False)


# RANKING FUNCTIONS ############################################################


    def rank_citation_pairs(self, citing, cited):

        # compute ranks (the ranks are just the densities of the items)
        # the user-specific information is disregarded
        item_idxs = cited[0]
        ranks = self.popularities.index_select(0, item_idxs).squeeze()

        return ranks


# RUN NAME HELPERS #############################################################


    def get_net_name(self):
        return 'ItemPopularity'


    def get_run_name(self):
        return 'default'

