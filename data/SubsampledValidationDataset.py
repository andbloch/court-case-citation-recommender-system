import os
import torch
from torch.utils.data import Dataset
import numpy as np
import pickle
import time
from .CaseQuintupleService import CaseQuintupleService


class SubsampledValidationDataset(Dataset):

    def __init__(self,
                 dataset_owner,
                 dataset_name,
                 dataset_type,
                 num_evaluation_examples,
                 net_name):
        """

        :param dataset_owner:
        :param dataset_name:
        :param dataset_type:
        :param num_evaluation_examples:
        """

        # initialize dataset superclass
        super(Dataset, self).__init__()

        # keep track of net name
        self.net_name = net_name

        # determine directory of dataset
        self.data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                dataset_owner,
                                dataset_name)

        # get positive examples
        self.data = torch.from_numpy(
            np.load(os.path.join(self.data_dir, dataset_type+'_data.npy'))
        ).type('torch.LongTensor')

        # set-up quintuple service
        self.quintuple_service = CaseQuintupleService()

        # store number of examples to evaluate
        self.num_evaluation_examples = num_evaluation_examples

        # get dataset properties
        properties_file = os.path.join(self.data_dir,
                                       'dataset_properties.pickle')
        with open(properties_file, 'rb') as handle:
            properties = pickle.load(handle)
            self.num_items = properties['num_items']
            self.items = np.array(properties['items'])

        # get interacted items per user
        interacted_items = os.path.join(self.data_dir,
                                        dataset_type+'_cited_items.pickle')
        with open(interacted_items, 'rb') as handle:
            self.interacted_items = pickle.load(handle)

        # seed the generator differently every time to ensure different negative
        # examples in every epoch
        np.random.seed(int(time.time()))

    def __len__(self):

        return self.data.size()[0]

    def __getitem__(self, idx):

        # get user_id and item_id of positive example
        citing_id = self.data[idx,0].item()
        cited_pos_id = self.data[idx,1].item()

        # generate negative examples
        neg_item_ids = self._sample_negative_examples(citing_id, cited_pos_id)

        # create ordered list for evaluation
        # [user_id, pos_item_id, neg_item_id_(1), ..., neg_item_id_(n)]
        # where n = num_evaluation_examples-1
        evaluation_items = [citing_id, cited_pos_id] + neg_item_ids

        evaluation_set = []
        for case_id in evaluation_items:
            quintuple = None
            if self.net_name != 'ItemPopularity':
                quintuple = self.quintuple_service.get(case_id)
            else:
                quintuple = (torch.tensor(case_id), None, None, None, None)
            evaluation_set.append(quintuple)

        return evaluation_set

    def _sample_negative_examples(self, citing_id, cited_pos_id):

        # create list of negative examples
        neg_examples = []

        # for the desired amount of negative examples
        for i in range(self.num_evaluation_examples-1):
            # sample negative example
            neg_example = self._sample_negative_example(citing_id,
                                                        cited_pos_id,
                                                        neg_examples)
            # append negative example to list of negative examples
            neg_examples.append(neg_example)

        return neg_examples

    def _sample_negative_example(self, citing_id, cited_pos_id, sampled_items):

        neg_example = -1

        found = False
        while not found:
            # sample a potential negative example
            neg_example = np.random.choice(self.items)
            # test validity conditions
            not_interacted = neg_example not in self.interacted_items[citing_id]
            not_already_sampled = neg_example not in sampled_items
            not_the_positive_item = neg_example != cited_pos_id
            if not_interacted and not_already_sampled and not_the_positive_item:
                found = True

        return neg_example
