import os
import pandas as pd
import numpy as np
import pickle


# DETERMINE PATHS  #############################################################


# determine most important directories
CURR_DIR = os.path.dirname(os.path.realpath(__file__))
DATA_DIR = os.path.join(
    CURR_DIR,
    '../CourtCases/default/raw',
)
OUTPUT_DIR = os.path.join(
    DATA_DIR, '..'
)

# determine input files
CITATIONS_PATH = os.path.join(OUTPUT_DIR, 'citations.pkl')
QUADRUPLES_PATH = os.path.join(OUTPUT_DIR, 'case_quadruples.pkl')

# determine output files
CITATIONS_INT_PATH = os.path.join(OUTPUT_DIR, 'citations_int.npy')
QUINTUPLES_INT_PATH = os.path.join(OUTPUT_DIR, 'case_quintuples_int.pkl')


# READ INPUT FILES #############################################################


print('\n---Reading Input Files---')

df = pd.read_pickle(CITATIONS_PATH)

quadruples = None
with open(QUADRUPLES_PATH, 'rb') as handle:
    quadruples = pickle.load(handle)

print('Done.')


# CREATE MAPPING: CASE_ID -> INTEGER ###########################################


print('\n---Creating Maping: (Case_ID -> Int)---')

a = df.citing_id.unique()
b = df.cited_id.unique()
c = np.concatenate((a,b))
df2 = pd.DataFrame(data=c, columns=['case_ids'])
all_case_ids = df2.case_ids.unique()

case_id_2_int = {}
current_int_id = 0
for i in range(all_case_ids.shape[0]):
    case_id = all_case_ids[i]
    case_id_2_int[case_id] = current_int_id
    current_int_id += 1

print('Done.')


# INTEGERIZE CITATION CASE_IDS #################################################


print('\n---Integerize Citation Case IDs---')


df.citing_id = df.citing_id.apply(lambda x: case_id_2_int[x])
df.cited_id = df.cited_id.apply(lambda x: case_id_2_int[x])

# remove word-count columns
df = df[['citing_id', 'cited_id']]

print('Done.')


# INTEGERIZE QUADRUPLE CASE_IDS ################################################


print('\n---Integerize Quadruple Case IDs---')

quintuples_int = {}

for key, value in quadruples.items():
    int_key = case_id_2_int[key]
    quintuples_int[int_key] = (int_key,) + value

print('Done.')


# SAVE INTEGERIZED CITATIONS AND QUADRUPLES ####################################


print('\n---Saving Citations and Case Quintuples---')

# store citations
np.save(CITATIONS_INT_PATH, df.values)

# store quadruples
with open(QUINTUPLES_INT_PATH, 'wb') as handle:
    pickle.dump(quintuples_int, handle, protocol=pickle.HIGHEST_PROTOCOL)

print('Done.')
