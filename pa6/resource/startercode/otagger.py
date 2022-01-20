import numpy as np
from hmm import HMM
from collections import defaultdict


def model_training(train_data, tags):
    """
    Train an HMM based on training data

    Inputs:
    - train_data: (1*num_sentence) a list of sentences, each sentence is an object of the "line" class
            defined in data_process.py (read the file to see what attributes this class has)
    - tags: (1*num_tags) a list containing all possible POS tags

    Returns:
    - model: an object of HMM class initialized with parameters (pi, A, B, obs_dict, state_dict) calculated
            based on the training dataset
    """

    # unique_words.keys() contains all unique words
    unique_words = get_unique_words(train_data)

    word2idx = {}
    tag2idx = dict()
    S = len(tags)
    ###################################################
    # TODO: build two dictionaries
    #   - from a word to its index
    #   - from a tag to its index
    # The order you index the word/tag does not matter,
    # as long as the indices are 0, 1, 2, ...
    ###################################################
    unique_words_list = []
    for word in unique_words.keys():
        unique_words_list.append(word)

    for ix in range(len(unique_words)):
        word2idx[unique_words_list[ix]] = ix

    for ix in range(len(tags)):
        tag2idx[tags[ix]] = ix

    pi = np.zeros(S)
    A = np.zeros((S, S))
    B = np.zeros((S, len(unique_words)))
    ###################################################
    # TODO: estimate pi, A, B from the training data.
    #   When estimating the entries of A and B, if
    #   "divided by zero" is encountered, set the entry
    #   to be zero.
    ###################################################
    cnt_pi = np.zeros(S)
    cnt_s = np.zeros(S)
    cnt_s1s2 = np.zeros((S, S))
    cnt_sx = np.zeros((S, len(unique_words)))

    for line in train_data:
        sent_len = line.length
        sentence = line.words
        tags = line.tags
        for i in range(sent_len):
            if sent_len == 1:
                first_x = sentence[i]
                first_s = tags[i]

                first_xi = word2idx[first_x]
                first_si = tag2idx[first_s]
                cnt_sx[first_si, first_xi] += 1
                cnt_s[first_si] += 1
                cnt_pi[first_si] += 1

            if i + 1 < sent_len:
                j = i + 1
                first_x = sentence[i]
                first_s = tags[i]
                sec_x = sentence[j]
                sec_s = tags[j]

                first_xi = word2idx[first_x]
                first_si = tag2idx[first_s]
                sec_xi = word2idx[sec_x]
                sec_si = tag2idx[sec_s]

                cnt_s1s2[first_si, sec_si] += 1
                cnt_sx[first_si, first_xi] += 1
                cnt_s[first_si] += 1
                if i == 0:
                    cnt_pi[first_si] += 1
                if j == sent_len - 1:
                    cnt_s[sec_si] += 1
                    cnt_sx[sec_si, sec_xi] += 1

    for s in range(S):
        pi[s] = cnt_pi[s] / len(train_data)

    for s1 in range(S):
        for s2 in range(S):
            A[s1, s2] = cnt_s1s2[s1, s2] / cnt_s[s1]

    for s in range(S):
        for x in range(len(unique_words)):
            B[s, x] = cnt_sx[s, x] / cnt_s[s]


    # DO NOT MODIFY BELOW
    model = HMM(pi, A, B, word2idx, tag2idx)
    return model


def sentence_tagging(test_data, model, tags):
    """
    Inputs:
    - test_data: (1*num_sentence) a list of sentences, each sentence is an object of the "line" class
    - model: an object of the HMM class
    - tags: (1*num_tags) a list containing all possible POS tags

    Returns:
    - tagging: (num_sentence*num_tagging) a 2D list of output tagging for each sentences on test_data
    """
    tagging = []
    ######################################################################
    # TODO: for each sentence, find its tagging using Viterbi algorithm.
    #    Note that when encountering an unseen word not in the HMM model,
    #    you need to add this word to model.obs_dict, and expand model.B
    #    accordingly with value 1e-6.
    ######################################################################
    word_ix = len(model.obs_dict)
    S = len(model.pi)
    add_val = np.ones((S, 1)) * 1e-6

    for line in test_data:
        for i in range(line.length):
            word = line.words[i]
            if word not in model.obs_dict:
                model.obs_dict[word] = word_ix
                word_ix += 1
                model.B = np.append(model.B, add_val, axis=1)

    for line in test_data:
        path = model.viterbi(line.words)
        tagging.append(path)

    return tagging

# DO NOT MODIFY BELOW
def get_unique_words(data):

    unique_words = {}

    for line in data:
        for word in line.words:
            freq = unique_words.get(word, 0)
            freq += 1
            unique_words[word] = freq

    return unique_words
