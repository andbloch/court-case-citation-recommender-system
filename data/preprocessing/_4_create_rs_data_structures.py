import os
import sys
import time
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm
import builtins as __builtin__


# PREPROCESSING PARAMETERS #####################################################

# 'CourtCases'
# 'default'

DATASET_OWNER = 'CourtCases'
DATASET_NAME = 'reduced'
MIN_CITATIONS_IN_CITING_CASE = 1
MIN_CITATIONS_OF_CASE = 1
SHOW_PLOTS = False


# FIXING PRINT ORDERING ########################################################


def print(s):
    time.sleep(0.1)
    sys.stdout.flush()
    __builtin__.print(s)
    sys.stdout.flush()
    time.sleep(0.1)


# DETERMINE DATA DIRECTORY #####################################################


# determine script context directory
DATA_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                         '..',
                         DATASET_OWNER,
                         DATASET_NAME)

# joins path name data directory
def jd(filename):
    return os.path.join(DATA_PATH, filename)


# DETERMINE OUTPUT AND SET UP PLOTTING AND ITS DIRECTORY #######################


# determine and create plot directory (if necessary)
PLOT_DIR = jd('plots_and_output')
if not os.path.exists(PLOT_DIR):
    os.makedirs(PLOT_DIR)


PLOT_COUNT = 1
def show_and_save_plot(plt, plot_filename):
    global PLOT_COUNT
    plt.savefig(os.path.join(PLOT_DIR, str(PLOT_COUNT) + '_' + plot_filename))
    if SHOW_PLOTS:
        plt.show()
    plt.close()
    PLOT_COUNT += 1


# REDIRECT STOUT ###############################################################


# (we can't use a logger as the progress bars won't be printed)
sys.stdout = open(os.path.join(PLOT_DIR, 'console_output.txt'), 'w')


# PRINT PARAMETERS #############################################################


def pretty_print(variable):
    print(variable + '=' + repr(eval(variable)))


print('\n---Parameters---')
pretty_print('DATASET_OWNER')
pretty_print('DATASET_NAME')
pretty_print('MIN_CITATIONS_IN_CITING_CASE')
pretty_print('MIN_CITATIONS_OF_CASE')


# DETERMINE FILE LOCATIONS #####################################################


# get citations file
CITATIONS_FILE = jd('citations_int.npy')

# determine written files:
DATASET_PROPERTIES_FILE = jd('dataset_properties.pickle')

TRAINING_SET_FILE = jd('training_data.npy')
TRAINING_CITED_ITEMS_FILE = jd('training_cited_items.pickle')
TRAINING_RANKED_ITEMS_FILE = jd('training_ranked_items.npy')
TRAINING_POPULARITIES_FILE = jd('training_popularities.npy')

VALIDATION_SET_FILE = jd('validation_data.npy')
VALIDATION_CITED_ITEMS_FILE = jd('validation_cited_items.pickle')
VALIDATION_RANKED_ITEMS_FILE = jd('validation_ranked_items.npy')
VALIDATION_POPULARITIES_FILE = jd('validation_popularities.npy')

TEST_SET_FILE = jd('test_data.npy')
TEST_CITED_ITEMS_FILE = jd('test_cited_items.pickle')
TEST_RANKED_ITEMS_FILE = jd('test_ranked_items.npy')
TEST_POPULARITIES_FILE = jd('test_popularities.npy')


# READING OF ORIGINAL RATINGS ##################################################


# define dataframe column names
COLUMN_NAMES = ['citing_id', 'cited_id'] #, 'citing_timestamp', 'cited_timestamp']

# read csv data into data frame
df = pd.DataFrame(data=np.load(CITATIONS_FILE), columns=COLUMN_NAMES)


# DATA FRAME STATISTICS FUNCTIONS ##############################################


def get_contained_cases(df):
    """
    gets the set of contained cases (case_id of citing or cited) and its
    cardinality
    :param df:
    :return:
    """
    # determine set of citing and cited articles
    citing = set(df.citing_id.unique().tolist())
    cited = set(df.cited_id.unique().tolist())
    # build the union
    contained_cases = list(citing | cited)
    # determine the cardinality
    num_contained_cases = len(contained_cases)
    return contained_cases, num_contained_cases


def get_interaction_statistics(df, key):
    key_popularities = df[key].value_counts(sort=True, ascending=False).values
    s = key.partition('_')[0]
    d_min = np.min(key_popularities)
    d_median = np.median(key_popularities)
    d_mean = np.mean(key_popularities)
    d_max = np.max(key_popularities)
    return ('[%d, %d, %d, %d]' % (d_min, d_median, d_mean, d_max) +
           ' \tcitations per '+s+ ' [min, median, mean, max]')


def show_and_save_popularity_plot(df, key, plot_filename, extra_title=''):

    # get item popularities (item_id, count) from data in sorted order
    key_popularities = df[key].value_counts(sort=True, ascending=False).values

    key_min_dict = {
        'citing_id': MIN_CITATIONS_IN_CITING_CASE,
        'cited_id': MIN_CITATIONS_OF_CASE
    }

    # get first index where the item popularity is below the threshold
    cutoff_idx = np.argmin(key_popularities > key_min_dict[key])
    if np.min(key_popularities) < key_min_dict[key]:
        plt.axvline(x=cutoff_idx, color='black', linestyle=':')

    # show popularity distribution
    plt.plot(np.arange(0, df[key].nunique()), key_popularities)
    plt.axhline(y=key_min_dict[key], color='black', linestyle=':')
    plt.xlim([0, len(key_popularities)])
    plt.ylim([0, np.max(key_popularities)])

    # get title
    title_dict = {
        'citing_id': 'in Citing Article',
        'cited_id': 'of Cited Article'
    }
    plt.title('#Citations '+title_dict[key]+' (Descending) '+ extra_title)

    # show and save plot
    show_and_save_plot(plt, plot_filename)


# READING AND PLOTTING OF DATASET STATISTICS ###################################


# determine original number of cases and citations
NUM_CASES_ORIG = get_contained_cases(df)[1]
NUM_CITATIONS_ORIG = df.shape[0]
NUM_CITING_ORIG = df.citing_id.nunique()
NUM_CITED_ORIG = df.cited_id.nunique()

# print original datset statistics
print('\n---Original Dataset---')
print('#cases: \t' + str(get_contained_cases(df)[1]))
print('#citations: ' + str(NUM_CITATIONS_ORIG))
citing_id_stats = get_interaction_statistics(df, 'citing_id')
print('#citing: \t' + str(NUM_CITING_ORIG) + ' \t' + citing_id_stats)
cited_id_stats = get_interaction_statistics(df, 'cited_id')
print('#cited: \t' + str(NUM_CITED_ORIG) + ' \t' + cited_id_stats)

# show popularity plots
show_and_save_popularity_plot(df, 'citing_id', 'citing_id_popularity.png')
show_and_save_popularity_plot(df, 'cited_id', 'cited_id_popularity.png')


# DATA REDUCTION PRINTING FUNCTIONS ############################################


def print_reduction_factors(df, title):

    # determine number of lost citations (and percentage)
    citations_lost = NUM_CITATIONS_ORIG - df.shape[0]
    citations_lost_pct = (citations_lost / NUM_CITATIONS_ORIG) * 100.0

    # determine number of lost citing articles (and percentage)
    num_citing_now = df.citing_id.nunique()
    citing_lost = NUM_CITING_ORIG - num_citing_now
    citing_lost_pct = (citing_lost / NUM_CITING_ORIG) * 100.0

    # determine number of lost citations (and percentage)
    num_cited_now = df.cited_id.nunique()
    cited_lost = NUM_CITED_ORIG - num_cited_now
    cited_lost_pct = (cited_lost / NUM_CITED_ORIG) * 100.0

    # get interaction statistics
    citing_id_stats = get_interaction_statistics(df, 'citing_id')
    cited_id_stats = get_interaction_statistics(df, 'cited_id')

    # print number of lost citing and cited articles
    print('\n---'+title+'---')
    print('#cases: \t' + str(get_contained_cases(df)[1]))
    print('#citations: ' + str(df.shape[0]) +
          (' \t (-%0.2f%%)' % citations_lost_pct))
    print('#citing: \t' + str(num_citing_now) +
          ' \t' + citing_id_stats + (' \t (-%.2f%%)' % citing_lost_pct))
    print('#cited: \t' + str(num_cited_now) +
          ' \t' + cited_id_stats + (' \t (-%.2f%%)' % cited_lost_pct))


# DATA REDUCTION (MIN CITATIONS) ###############################################


# further ensure that only citing and cited articles with specified thresholds
# are considered.
changes = True
df_prev_size = df.shape[0]
while changes:
    # count appearances of citing articles
    vc = df.citing_id.value_counts()
    # only keep citing articles which do specified amount of minimal citations
    threshold = MIN_CITATIONS_IN_CITING_CASE - 1
    df = df[df.citing_id.isin(vc.index[vc.gt(threshold)])]
    # count appearances of cited articles
    vc = df.cited_id.value_counts()
    # only keep cited articles which have specified amount of minimal citations
    threshold = MIN_CITATIONS_OF_CASE - 1
    df = df[df.cited_id.isin(vc.index[vc.gt(threshold)])]
    # check current data frame size
    df_curr_size = df.shape[0]
    # check if changes to the data frame size happened
    if df_curr_size < df_prev_size:
        changes = True  # (changes occurred)
    else:
        changes = False # (no changes occured, and non will in the future)
    # current dataframe size
    df_prev_size = df_curr_size

# print reduction factors
print_reduction_factors(df, 'Reduced Dataset (Min Citations per/of Case)')

# ensure that every case has enough citations done/recevied
assert(df.citing_id.value_counts().min() >= MIN_CITATIONS_IN_CITING_CASE)
assert(df.cited_id.value_counts().min() >= MIN_CITATIONS_OF_CASE)

# show popularity plots
show_and_save_popularity_plot(df, 'citing_id', 'citing_id_popularity.png',
                              extra_title='after #Citations Reduction')
show_and_save_popularity_plot(df, 'cited_id', 'cited_id_popularity.png',
                              extra_title='after #Citations Reduction')


# STORING OF DATASET PROPERTIES ################################################


# determine final number of cases
CASES, NUM_CASES = get_contained_cases(df)

# store the data set properties
properties = {
    'num_items': NUM_CASES,
    'items': CASES
}
with open(DATASET_PROPERTIES_FILE, 'wb') as handle:
    pickle.dump(properties, handle, protocol=pickle.HIGHEST_PROTOCOL)


# CITATION TIME DISTANCE STATISTICS ############################################


if False:
    # compute citation time differences
    df['citation_time_diff'] = \
        df['citing_timestamp'].sub(df['cited_timestamp'], axis=0)

    # get citation time differences
    citation_time_differences = df['citation_time_diff'].values

    # compute relevant percentiles
    pct5 = np.percentile(citation_time_differences, 5)
    pct90 = np.percentile(citation_time_differences, 90)

    # use percentiles as parameters
    VALIDATION_TIMESPAN = 0.5 * pct5    # TODO: maybe pct90 not needed
    TRAINING_START_TIMESPAN = pct90     # TODO: think about this

    # pretty-print validation timespan
    pretty_print('VALIDATION_TIMESPAN')


    def create_citation_time_difference_histogram(df):
        """
        shows distribution of time differences
        :param df:
        :return:
        """

        # determine maximal possible timespan in data frame
        max_time = df['citing_timestamp'].max()
        min_time = df['citing_timestamp'].min()
        max_timespan = max_time - min_time

        # get time differences
        time_differences = df['citation_time_diff']

        # show test and validation cutoffs
        plt.axvline(x=VALIDATION_TIMESPAN, color='grey', linestyle=':')
        plt.axvline(x=2*VALIDATION_TIMESPAN, color='black', linestyle=':')
        plt.axvline(x=TRAINING_START_TIMESPAN, color='blue', linestyle=':')

        # plot histogram
        resolution = 50
        plt.hist(time_differences.values, resolution, density=True, facecolor='g')

        # set axes
        #plt.xlim([0, max_timespan])
        #plt.ylim([0, 1])
        # add title
        plt.title('Citation Time Difference ')

        # show plot
        if SHOW_PLOTS:
            plt.show()


    # show citation_time_difference_histogram
    create_citation_time_difference_histogram(df)

    # drop citation_time_diff column
    df = df.drop(columns=['citation_time_diff'])


# TRAINING, VALIDATION, TEST SPLITTING #########################################


print('\n---Building Training, Validation and Test Set from '+\
      'Positive Examples---')

# determine set of citing cases
NUM_CITATIONS = df.shape[0]
citing_cases = df.citing_id.unique()
num_citing_cases = citing_cases.shape[0]

print('Splitting ' + str(num_citing_cases) + ' cases with ' +
      str(NUM_CITATIONS)+ ' citations into:')

# shuffle citing cases
np.random.seed(42)  # repeatable randomness!!!
np.random.shuffle(citing_cases)

# pick first 10% as test cases, next 10% as validation cases, rest for training
n_10pct = int(num_citing_cases*0.1)
test_cases = citing_cases[0:n_10pct]
validation_cases = citing_cases[n_10pct:(2*n_10pct)]
training_cases = citing_cases[(2*n_10pct):]

# assert that the number of cases in each set
assert(num_citing_cases == test_cases.shape[0] +
       validation_cases.shape[0] +
       training_cases.shape[0])

# create training, validation and test data frames
df_training = df[df.citing_id.isin(training_cases)]
df_validation = df[df.citing_id.isin(validation_cases)]
df_test = df[df.citing_id.isin(test_cases)]

# assert that the number of citations adds up to the original number of citatins
# of the reduced data
n_rows_now = df_training.shape[0] + df_validation.shape[0] + df_test.shape[0]
assert(n_rows_now == NUM_CITATIONS)

# save training, validation and test set
np.save(TRAINING_SET_FILE, df_training.values)
np.save(VALIDATION_SET_FILE, df_validation.values)
np.save(TEST_SET_FILE, df_test.values)

print('///Training///')
print('#cases: ' + str(training_cases.shape[0]))
print('#citations: ' + str(df_training.shape[0]))
print(get_interaction_statistics(df_training, 'citing_id'))
print(get_interaction_statistics(df_training, 'cited_id'))

print('///Validation///')
print('#cases: ' + str(validation_cases.shape[0]))
print('#citations: ' + str(df_validation.shape[0]))
print(get_interaction_statistics(df_validation, 'citing_id'))
print(get_interaction_statistics(df_validation, 'cited_id'))

print('///Test///')
print('#cases: ' + str(test_cases.shape[0]))
print('#citations: ' + str(df_test.shape[0]))
print(get_interaction_statistics(df_test, 'citing_id'))
print(get_interaction_statistics(df_test, 'cited_id'))

# report density_percent of training data (positive and negative examples)
num_experiences = df_training.shape[0]
density_percent = (num_experiences / (NUM_CASES * NUM_CASES)) * 100.0
print('Density of training data: \t%.5f%%' % density_percent)


# CREATE DICT OF CITED CASES ###################################################


def create_cited_cases_dict(df, dict_filename):
    """
    Creates a dictionary with the following mapping:
                citing_id |--> set of cited_ids
    such that the cited cases can also be excluded when doing a prediction
    with a model that was trained on the examples.
    :param df:
    :param dict_filename:
    :return:
    """

    # determine list of citing cases
    citing_ids = df.citing_id.unique().tolist()

    # create empty dictionary for each citing article
    cited_articles = {citing_id: set() for citing_id in citing_ids}

    # iterate over rows
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        # add citation to cited cases
        citing_id = row['citing_id']
        cited_id = row['cited_id']
        cited_articles[citing_id].add(cited_id)

    # pickle cited articles dict
    with open(dict_filename, 'wb') as handle:
        pickle.dump(cited_articles, handle,
                    protocol=pickle.HIGHEST_PROTOCOL)


print('\n---Creating Cited Cases Dicts---')

# create training cited cases
create_cited_cases_dict(df_training,
                        TRAINING_CITED_ITEMS_FILE)

# create validation cited cases
df_training_validation = pd.concat([df_training, df_validation])
create_cited_cases_dict(df_training_validation,
                        VALIDATION_CITED_ITEMS_FILE)

# create test cited cases
df_training_validation_test = pd.concat([df_training, df_validation, df_test])
create_cited_cases_dict(df_training_validation_test,
                        TEST_CITED_ITEMS_FILE)

print('Done.')


# ITEM-POPULARITY MODELS #######################################################


# note that the popularities are based on ANY ranking (1 to 5)

def create_popularity_model(df, ranks_filename, popularities_filename):

    # get cited items (item_id, count) ranked (and sorted) by their popularity
    ranked_items = df.cited_id.value_counts(sort=True, ascending=False)

    # add 0 counts for items that do not appear in data
    missing_items = {}
    for item_id in properties['items']:
        if item_id not in ranked_items.index:
            missing_items[item_id] = 0
    if len(missing_items) > 0:
        df_missing_items = pd.DataFrame.from_dict(missing_items, orient='index')
        ranked_items = pd.concat([ranked_items, df_missing_items])

    # create vector containing items in order by their ranking (popularity)
    items_ordered_by_rank = ranked_items.index.values

    # write ranks to disk
    with open(ranks_filename, 'wb') as handle:
        pickle.dump(items_ordered_by_rank, handle,
                    protocol=pickle.HIGHEST_PROTOCOL)

    # crate index (item_id -> popularity (density))
    popularities = ranked_items.sort_index().values
    # normalize popularities
    popularities = popularities / np.sum(popularities)

    # write popularities to disk
    with open(popularities_filename, 'wb') as handle:
        pickle.dump(popularities, handle, protocol=pickle.HIGHEST_PROTOCOL)


print('\n---Creating Item Popularities---')

# create training popularity model
create_popularity_model(df_training,
                        TRAINING_RANKED_ITEMS_FILE,
                        TRAINING_POPULARITIES_FILE)

# create validation popularity model
create_popularity_model(df_validation,
                        VALIDATION_RANKED_ITEMS_FILE,
                        VALIDATION_POPULARITIES_FILE)

# create test popularity model
create_popularity_model(df_test,
                        TEST_RANKED_ITEMS_FILE,
                        TEST_POPULARITIES_FILE)

print('Done.')



