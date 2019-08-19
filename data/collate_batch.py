import os
import torch
import pickle

# read word to index mapping
WORD2IDX_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                            '../pretrained_word_embedding/glove.6B/word2idx.pkl')

word2idx = pickle.load(open(WORD2IDX_DIR, 'rb'))

# define collate_fn
def collate_batch_examples(batch, device, net_name):

    # TOOD: pre-batch to 'device' once you know how

    # determine batch size
    batch_size = len(batch)

    # create list containing the 3-tuples
    # (D_padded, document_lengths, sentence lengths)
    # for each i-th example (citing, cited, neg_1, neg_2, ..., neg_K)
    citing_cited_negatives = []

    # for each i-th example in a batch item
    num_items_per_batch_example = len(batch[0])
    for i in range(num_items_per_batch_example):

        # get list of i-th 5-tuple of each batch item
        document_5tuples = [list_of_5tuples[i] for list_of_5tuples in batch]

        # decompose 5 tuples:
        # get list of case_ids
        case_ids = [_5tuple[0] for _5tuple in document_5tuples]
        case_ids = torch.LongTensor(case_ids)

        # performance optimization for ItemPopularity model
        if net_name != 'ItemPopularity':

            # get list of documents (list of lists of integers)
            D = [_5tuple[1] for _5tuple in document_5tuples]
            # determine document and sentence lengths
            num_sentences = [_5tuple[3] for _5tuple in document_5tuples]
            sentence_lengths = [_5tuple[2] for _5tuple in document_5tuples]
            max_sentence_lengths = [_5tuple[4] for _5tuple in document_5tuples]

            # prepare document tensor D of right dimensions:
            # (num_documents, max_num_sentences, max_max_num_words)
            num_documents = batch_size
            max_num_sentences = max(num_sentences)
            max_max_num_words = max(max_sentence_lengths)
            dims = (num_documents, max_num_sentences, max_max_num_words)
            # create padded tensor D (initialized with pad value)
            D_padded = torch.full(dims, word2idx['<PAD>'], dtype=torch.int64)

            # copy over the actual sequences (word indices) of each document into
            # padded tensor D_padded:
            # for each document j
            for j in range(num_documents):
                # for each sentence k in document j
                for k in range(num_sentences[j]):
                    # determine length of k-th sentence
                    sentence_length = sentence_lengths[j][k]
                    # copy the contents of the k-th sentence into the padded matrix
                    # (D[j][k] is list of integers)
                    D_padded[j,k,0:sentence_length] = torch.LongTensor(D[j][k])

            # append triplet (use list to support re-assignment)
            citing_cited_negatives.append([case_ids,
                                           D_padded,
                                           num_sentences,
                                           sentence_lengths])
        else:
            # append triplet (use list to support re-assignment)
            citing_cited_negatives.append([case_ids,
                                           None,
                                           None,
                                           None])

    return citing_cited_negatives