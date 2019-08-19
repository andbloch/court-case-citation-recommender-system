import os
import torch
from torch.utils.data import Dataset
import numpy as np
import pickle
import time
from .CaseQuintupleService import CaseQuintupleService

class TripletDataset(Dataset):


    def __init__(self,
                 dataset_owner,
                 dataset_name,
                 dataset_type,
                 num_negative_examples,
                 net_name):

        # initialize dataset superclass
        super(TripletDataset, self).__init__()

        # keep track of net name
        self.net_name = net_name

        # determine directory of dataset
        self.data_dir = \
            os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                         dataset_owner,
                                         dataset_name)

        # get positive training examples as table
        self.data = torch.from_numpy(
            np.load(os.path.join(self.data_dir, dataset_type+'_data.npy'))
        ).type('torch.LongTensor')

        # set-up quintuple service
        self.quintuple_service = CaseQuintupleService()

        # get dataset properties
        properties_file = os.path.join(self.data_dir,
                                       'dataset_properties.pickle')
        with open(properties_file, 'rb') as handle:
            properties = pickle.load(handle)
            self.num_items = properties['num_items']
            self.items = np.array(properties['items'])

        # store negative sampling factor
        self.K = num_negative_examples

        # get positive example dict (citing_id -> set of cited articles)
        positive = os.path.join(self.data_dir,
                                dataset_type+'_cited_items.pickle')
        with open(positive, 'rb') as handle:
            self.positive_items = pickle.load(handle)

        # seed the generator differently every time to ensure different negative
        # examples in every epoch
        np.random.seed(int(time.time()))


    def __len__(self):

        return self.data.size()[0]*self.K


    def __getitem__(self, idx):

        # get index of real example
        idx_real = int(np.floor(idx / self.K))

        # get citing_id and cited_id of positive example
        citing_id = self.data[idx_real,0].item()
        cited_pos_id = self.data[idx_real,1].item()

        # sample/generate a negative example
        # (an item that the user hasn't interacted with)
        cited_neg_id = self._sample_negative_example(citing_id)

        # retrieve quintuples
        citing_quintuple = None
        cited_pos_quintuple = None
        cited_neg_quintuple = None
        # performance optimization for ItemPopularity model
        if self.net_name != 'ItemPopularity':
            citing_quintuple = self.quintuple_service.get(citing_id)
            cited_pos_quintuple = self.quintuple_service.get(cited_pos_id)
            cited_neg_quintuple = self.quintuple_service.get(cited_neg_id)
        else:
            citing_quintuple = \
                (self.data[idx_real,0], None, None, None, None)
            cited_pos_quintuple = \
                (self.data[idx_real,1], None, None, None, None)
            cited_neg_quintuple = \
                (torch.tensor(cited_neg_quintuple), None, None, None, None)

        # create a triplet for BPR learning
        triplet = [citing_quintuple, cited_pos_quintuple, cited_neg_quintuple]

        return triplet


    def _sample_negative_example(self, citing_id):

        negative_example = -1

        found = False
        while not found:
            # sample a potential negative example
            negative_example = np.random.choice(self.items)
            # test if user has not interacted with that example
            if negative_example not in self.positive_items[citing_id]:
                found = True

        return negative_example
