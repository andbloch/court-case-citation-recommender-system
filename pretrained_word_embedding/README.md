**Word Embedding**

We use the pre-trained GloVE word embeddings that have been trained on 
Wikipedia 2014 and Gigaword 5. We'll use smallest embedding size of
50 dimensions, stored in "glove.6B.50d.txt". It has a vocabulary size of
exactly 400'000 words.

You may download glove.6B.zip from here: 

http://nlp.stanford.edu/data/wordvecs/glove.6B.zip

Save and extract the ZIP file in this folder such that glove is stored in the directory
"glove.6B". Then run the file 

"create_embedding_data_structures.py" 

to generate the necessary data-structures for the vocab. and word2idx/idx2vec 
mappings.