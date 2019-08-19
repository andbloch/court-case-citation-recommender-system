import os
import pandas as pd
import pickle
import psycopg2
from tqdm import tqdm
from data.preprocessing.util.text_processing import process_text


# LOAD CITATIONS.PKL ###########################################################


print('\n---Reading Citations Data Frame---')

# determine most important directories
CURR_DIR = os.path.dirname(os.path.realpath(__file__))
DATA_DIR = os.path.join(
    CURR_DIR,
    '../CourtCases/default/raw',
)
OUTPUT_DIR = os.path.join(
    DATA_DIR, '..'
)
CITATIONS_PATH = os.path.join(OUTPUT_DIR, 'citations.pkl')
QUADRUPLES_PATH = os.path.join(OUTPUT_DIR, 'case_quadruples.pkl')

df = pd.read_pickle(CITATIONS_PATH)

print('Done.')


# DB ACCESS FUNCTIONS ##########################################################

# ssh -N -f -L localhost:6666:localhost:32385 abloch@login.leonhard.ethz.ch
# TODO: input database credentials if you want to access the database!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# TODO: input database credentials if you want to access the database!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# TODO: input database credentials if you want to access the database!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# TODO: input database credentials if you want to access the database!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# TODO: input database credentials if you want to access the database!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# open db connection
conn = psycopg2.connect(database='',
                        user='',
                        password='',
                        host='localhost',
                        port=6666)

def get_case_data(citing_id):
    """
    - opinion: the case's text
    - judges: the judges treating the case currently
    - history judges: the judges treating the case previously
    - dc identifier: unique court-case identifier from Lexis DB
    - date standard: contains one date from the previous dates based on the
      following priorities: decided, filed, revised, argued, unspecified, values
      range from 1658 - 2017
    - word count: num words in the case
    """

    # create a new cursor
    cursor = conn.cursor()

    # get all circuit cases
    # with more than 'MIN_TEXT_SIZE' words
    # in a random (repeatable) order
    query = \
        "SELECT op.opinion, cc.judges, cc.history_judges " +\
        "FROM lexis_opinions_circuit AS op " + \
        "JOIN lexis_cases_circuit AS cc USING(case_type, dc_identifier) " + \
        "WHERE op.dc_identifier='"+citing_id+"' " + \
        "LIMIT 1;"
    cursor.execute(query)

    # get SQL result
    sql_row = cursor.fetchone()

    # close the cursor
    cursor.close()

    if sql_row is not None:
        # get opinion text
        opinion_text = sql_row[0]
        if opinion_text is not None:
            # get judges data
            judges = []
            judges1 = sql_row[1]
            judges2 = sql_row[2]
            if judges1 is not None:
                judges.extend(judges1)
            if judges2 is not None:
                judges.extend(judges2)
            # create case data
            case_data = {
                'opinion_text': opinion_text,
                'judges': judges
            }
            return case_data
    return None


# CREATE QUADRUPLES DICTIONARY #################################################


print('\n---Generating Citation Quadruples---')


# create quadruples dictionary
quadruples_dict = {}


def add_quadruple(case_id):
    global quadruples_dict
    case_data = get_case_data(case_id)
    if case_data is not None:
        integerized_text = process_text(case_data['opinion_text'],
                                        case_data['judges'])

        # determine sentence lengths
        sentence_lengths = [len(sentence) for sentence in integerized_text]
        # determine number of sentences
        num_sentences = len(sentence_lengths)
        # determine length of longest sentence
        max_sentence_length = max(sentence_lengths)
        # build document quadruple
        quadruple = (integerized_text,
                     sentence_lengths,
                     num_sentences,
                     max_sentence_length)
        # add document quadruple to quadruple dictionary
        quadruples_dict[case_id] = quadruple
    else:
        print(case_id)
        print('Not found!!')


# create quadruples for all citing and cited articles
for index, row in tqdm(df.iterrows(), total=df.shape[0]):
    if not row.citing_id in quadruples_dict.keys():
        add_quadruple(row.citing_id)
    if not row.cited_id in quadruples_dict.keys():
        add_quadruple(row.cited_id)


# SAVE CASE-QUADRUPLES DICTIONARY ##############################################


print('\n---Saving Case Quadruples File---')

# Store data (serialize)
with open(QUADRUPLES_PATH, 'wb') as handle:
    pickle.dump(quadruples_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

print('Done.')
