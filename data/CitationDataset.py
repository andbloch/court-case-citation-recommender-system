import os
import pickle
import numpy as np
import torch

from .TripletDataset import TripletDataset
from .SubsampledValidationDataset import SubsampledValidationDataset


# CITATION DATASET CLASS #######################################################


class CitationDataset(object):

    def __init__(self,
                 dataset_owner,
                 dataset_name,
                 loss_name,
                 num_negative_examples,
                 hit_rate_num_negative_examples,
                 net_name):

        # initialize object superclass
        super(CitationDataset, self).__init__()

        # keep track of supplied parameters
        self.dataset_owner = dataset_owner
        self.dataset_name = dataset_name
        self.loss_name = loss_name
        self.num_negative_examples = num_negative_examples
        self.hit_rate_num_negative_examples = hit_rate_num_negative_examples
        self.net_name = net_name

        # determine directory of dataset
        self.data_dir = \
            os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                         self.dataset_owner,
                                         self.dataset_name)

        # get dataset properties
        properties_file = os.path.join(self.data_dir,
                                       'dataset_properties.pickle')
        with open(properties_file, 'rb') as handle:
            properties = pickle.load(handle)
            self.num_items = properties['num_items']
            self.items = np.array(properties['items'])

    def _get_set(self, dataset_type):
        """returns a (single/multi-pair) dataset"""
        dataset = TripletDataset(self.dataset_owner,
                                 self.dataset_name,
                                 dataset_type,
                                 self.num_negative_examples,
                                 self.net_name)
        return dataset

    def _get_subsampled_set(self, dataset_type):
        """returns a subsampled dataset for HR computation"""
        subsampled_dataset = \
            SubsampledValidationDataset(self.dataset_owner,
                                        self.dataset_name,
                                        dataset_type,
                                        self.hit_rate_num_negative_examples,
                                        self.net_name)
        return subsampled_dataset

    def get_training_set(self):
        return self._get_set('training')

    def get_validation_set(self):
        return self._get_set('validation')

    def get_subsampled_validation_set(self):
        return self._get_subsampled_set('validation')

    def get_subsampled_test_set(self):
        return self._get_subsampled_set('test')

    def _get_ranked_items(self, dataset_type):
        # get rankings by popularity
        # (list of items ranked/ordered by their popularity)
        ranked_items = torch.from_numpy(
            np.load(os.path.join(self.data_dir,
                                 dataset_type + '_ranked_items.npy'))
        ).type('torch.LongTensor')
        return ranked_items

    def get_training_ranked_items(self):
        return self._get_ranked_items('training')

    def _get_popularities(self, dataset_type):
        # get popularities dict: item id -> popularity (density)
        popularities_file = os.path.join(self.data_dir,
                                         dataset_type + '_popularities.npy')
        popularities = None
        with open(popularities_file, 'rb') as handle:
            popularities = pickle.load(handle)

        popularities = torch.from_numpy(popularities).squeeze(-1)
        return popularities

    def get_training_popularities(self):
        return self._get_popularities('training')
