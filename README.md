# Recommender System for US Court Cases

This repository contains the prototype of a recommender system that can do
content-based citation-recommendations based on the majority opinion of
US court cases.

## Folder Structure

- `data`: Contains  all the code for 
  - Data preprocessing
  - Creation of the datastructures required for training, testing and 
validation
  - Data analysis
- `document_embedding`: Contains all the code for the HRNN document embedder
- `pretrained_word_embedding`: contains all the code to generate the needed
datastructures from the pre-trained GloVe word embedding
- `ranking_models`: Contains the code for the `ItemPopularity` and 
`EmbedTextNCF` ranking models
- `training`: Contains the code for the training of the recommender systems

## Running the Code

In order to run the entire training you must do the following steps

1) Create the word-embedding datastructures from GloVe as described in the
folder `pretrained_word_embedding`
2) Create the training, validaiton and test data structures as described in
the folder `data` (this requires access to a proprietary dataset from LawEcon)
