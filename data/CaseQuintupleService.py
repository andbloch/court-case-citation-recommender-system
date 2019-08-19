import os
import pickle

class CaseQuintupleService:
    """ A python singleton """

    def __load_quintuples(self):
        """ Implementation of the singleton interface """

        # determine directory of dataset
        data_dir = \
            os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                         'CourtCases',
                                         'default')

        # load case quintuples
        quintuples_path = os.path.join(data_dir, 'case_quintuples_int.pkl')
        with open(quintuples_path, 'rb') as handle:
            return pickle.load(handle)

    # storage for the instance reference
    quintuples = None

    def __init__(self):
        """ Create singleton instance """
        # Check whether we already have an instance
        if CaseQuintupleService.quintuples is None:
            # Create and remember instance
            CaseQuintupleService.quintuples = self.__load_quintuples()

    def get(self, case_id):
        return CaseQuintupleService.quintuples[case_id]
