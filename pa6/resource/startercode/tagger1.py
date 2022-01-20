import numpy as np

from util import accuracy
from hmm import HMM

# TODO:
def model_training(train_data, tags):
	"""
	Train HMM based on training data

	Inputs:
	- train_data: (1*num_sentence) a list of sentences, each sentence is an object of line class
	- tags: (1*num_tags) a list of POS tags

	Returns:
	- model: an object of HMM class initialized with parameters(pi, A, B, obs_dict, state_dict) you calculated based on train_data
	"""
	model = None
	###################################################
	N = len(tags)
	A = np.ones((N, N)) / N
	pi = np.ones(N) / N

	state_dict, tag_dict, obs_dict = {}, {}, {}
	word_list = []

	for idx, tag in enumerate(tags):
		state_dict[tag] = idx

	for cur_line in train_data:
		pi[state_dict[cur_line.tags[0]]] += 1
		for idx in range(cur_line.length):
			tag = cur_line.tags[idx]
			word_list.append(cur_line.words[idx])
			if tag not in tag_dict:
				tag_dict[tag] = 1
			else:
				tag_dict[tag] += 1
			if idx < cur_line.length - 1:
				A[tags.index(cur_line.tags[idx]), tags.index(cur_line.tags[idx + 1])] += 1

	word_list = list(set(word_list))
	for idx, word in enumerate(word_list):
		obs_dict[word] = idx

	total_tags = sum(tag_dict.values())
	for key in tag_dict.keys():
		tag_dict[key] /= total_tags

	B = np.zeros([N, len(word_list)])
	for line in train_data:
		for word, tag in zip(line.words, line.tags):
			B[state_dict[tag], obs_dict[word]] = tag_dict[tag]

	A /= np.sum(A, axis=1)[:, None]
	pi /= len(train_data)
	model = HMM(pi, A, B, obs_dict, state_dict)
	###################################################
	return model


def sentence_tagging(test_data, model, tags):
	"""
	Inputs:
	- test_data: (1*num_sentence) a list of sentences, each sentence is an object of line class
	- model: an object of HMM class

	Returns:
	- tagging: (num_sentence*num_tagging) a 2D list of output tagging for each sentences on test_data
	"""
	tagging = []
	###################################################
	N = len(model.pi)
	idx = len(model.obs_dict)
	prob = np.ones((N, 1)) * 1e-6
	for line in test_data:
		for i in range(len(line.words)):
			cur_word = line.words[i]
			if cur_word not in model.obs_dict:
				model.obs_dict[cur_word] = idx
				model.B = np.append(model.B, prob, axis=1)
				idx += 1

	for line in test_data:
		new_path = model.viterbi(line.words)
		tagging.append(new_path)
	###################################################
	return tagging
