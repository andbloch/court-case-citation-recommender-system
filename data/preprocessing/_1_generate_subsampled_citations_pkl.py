import os
import pandas as pd
import psycopg2
import numpy as np


# LOAD CITATION NETWORK ########################################################


print('\n---Loading Citation Network---')

CITATION_NETWORK_FILENAME = '20190505_case_citation_network.csv'
NUM_LINES_CITATION_NETWORK = 81770663
NUM_SUBSAMPLED_CASES = 7000
MIN_WORD_COUNT = 50

# determine most important directories
CURR_DIR = os.path.dirname(os.path.realpath(__file__))
DATA_DIR = os.path.join(
    CURR_DIR,
    '../CourtCases/default/raw',
)
OUTPUT_DIR = os.path.join(
    DATA_DIR, '..'
)

# determine citation network csv location
CITATION_NETWORK_PATH = os.path.join(DATA_DIR, CITATION_NETWORK_FILENAME)
print('Path: '+str(CITATION_NETWORK_PATH))

# define dataframe column names
COLUMN_NAMES = [
    'id',
    'citing_case_type',
    'citing_id',
    'cited_lexis_id',
    'cited_lexis_id_normalized',
    'citation_type',
    'most_probable_cited_case_type',
    'cited_id',
    # additional columns
    'citing_word_count',
    'cited_word_count',
]

# read citation network data into data frame
df = pd.read_csv(CITATION_NETWORK_PATH,
                 names=COLUMN_NAMES,
                 sep=',',
                 skiprows=1)
print('#rows: '+str(df.shape[0]))
print('Loaded.')


# SUBSAMPLE CITATION NETWORK ###################################################


print('\n---Subsampling Citation Network---')
print('Picking only Circuit Court cases.')
print('# Subsampled Cases: \t'+str(NUM_SUBSAMPLED_CASES))

# filter df by circuit case type
df = df[df.citing_case_type==1]

# remove unused columns
df.drop(columns=[
    'id',
    'citing_case_type',
    'cited_lexis_id',
    'cited_lexis_id_normalized',
    'citation_type',
    'most_probable_cited_case_type'
], inplace=True)

# shuffle data frame repeatably
np.random.seed(42)
df = df.reindex(np.random.permutation(df.index))

# get subsample of cases
taken_cases = df.citing_id.unique().tolist()[:NUM_SUBSAMPLED_CASES]

# filter data from down to taken cases
df = df[df.citing_id.isin(taken_cases)]

print('#rows: '+str(df.shape[0]))
print('Done.')


# GET WORD COUNTS ##############################################################


print('\n---Getting Word Counts---')

# ssh -N -f -L localhost:6666:localhost:32385 abloch@login.leonhard.ethz.ch
# open db connection
# TODO: input database credentials if you want to access the database!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# TODO: input database credentials if you want to access the database!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# TODO: input database credentials if you want to access the database!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# TODO: input database credentials if you want to access the database!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# TODO: input database credentials if you want to access the database!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
conn = psycopg2.connect(database='',
                        user='',
                        password='',
                        host='localhost',
                        port=6666)

def get_word_count(case_id):

    # create a new cursor
    cursor = conn.cursor()

    # get all circuit cases
    # with more than 'MIN_TEXT_SIZE' words
    # in a random (repeatable) order
    query = \
        "SELECT op.word_count " +\
        "FROM lexis_opinions_circuit AS op " + \
        "WHERE op.dc_identifier='" + case_id + "' " + \
        "LIMIT 1;"
    cursor.execute(query)

    # get SQL result
    sql_row = cursor.fetchone()

    # close the cursor
    cursor.close()

    if sql_row is not None:
        return sql_row[0]
    else:
        return None


# get word counts of citing and cited article
df.citing_word_count = df.citing_id.apply(get_word_count)
df.cited_word_count = df.cited_id.apply(get_word_count)

print('Done.')


# FILTER CITATIONS BY WORD-COUNT ###############################################


print('\n---Filtering by Word Count---')
print('Min word count: '+str(MIN_WORD_COUNT))

# filter citations by word count
df = df[df.citing_word_count >= MIN_WORD_COUNT]
df = df[df.cited_word_count >= MIN_WORD_COUNT]

print('#rows: '+str(df.shape[0]))
print('Done.')


# SAVE CITATIONS.PKL FILE ######################################################


print('\n---Saving \'citations.pkl\' File---')

CITATIONS_PATH = os.path.join(OUTPUT_DIR, 'citations.pkl')
df.to_pickle(CITATIONS_PATH)

print('Done.')