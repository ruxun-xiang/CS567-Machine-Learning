from __future__ import print_function
import numpy as np


class HMM:

    def __init__(self, pi, A, B, obs_dict, state_dict):
        """
        - pi: (1*num_state) A numpy array of initial probailities. pi[i] = P(Z_1 = s_i)
        - A: (num_state*num_state) A numpy array of transition probailities. A[i, j] = P(Z_t = s_j|Z_t-1 = s_i)
        - B: (num_state*num_obs_symbol) A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
        - obs_dict: (num_obs_symbol*1) A dictionary mapping each observation symbol to their index in B
        - state_dict: (num_state*1) A dictionary mapping each state to their index in pi and A
        """
        self.pi = pi
        self.A = A
        self.B = B
        self.obs_dict = obs_dict
        self.state_dict = state_dict

    def forward(self, Osequence):
        """
        Inputs:
        - self.pi: (1*num_state) A numpy array of initial probailities. pi[i] = P(Z_1 = s_i)
        - self.A: (num_state*num_state) A numpy array of transition probailities. A[i, j] = P(Z_t = s_j|Z_t-1 = s_i)
        - self.B: (num_state*num_obs_symbol) A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - alpha: (num_state*L) A numpy array alpha[i, t] = P(Z_t = s_i, x_1:x_t | λ)
        """
        S = len(self.pi)
        L = len(Osequence)
        alpha = np.zeros([S, L])
        ###################################################
        alpha[:, 0] = np.multiply(self.pi, self.B[:, self.obs_dict[Osequence[0]]])
        for t in range(1, L):
            for s in range(S):
                alpha[s, t] = np.multiply(self.B[s, self.obs_dict[Osequence[t]]], np.sum(np.matmul(alpha[:, t - 1], self.A[:, s])))
        ###################################################
        return alpha

    def backward(self, Osequence):
        """
        Inputs:
        - self.pi: (1*num_state) A numpy array of initial probailities. pi[i] = P(Z_1 = s_i)
        - self.A: (num_state*num_state) A numpy array of transition probailities. A[i, j] = P(Z_t = s_j|Z_t-1 = s_i)
        - self.B: (num_state*num_obs_symbol) A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - beta: (num_state*L) A numpy array beta[i, t] = P(x_t+1:x_T | Z_t = s_i, λ)
        """
        S = len(self.pi)
        L = len(Osequence)
        beta = np.zeros([S, L])
        ###################################################
        beta[:, L - 1] = 1
        for t in range(L - 2, -1, -1):
            for s in range(S):
                for _s in range(S):
                    beta[s, t] += beta[_s, t + 1] * self.A[s, _s] * self.B[_s, self.obs_dict[Osequence[t + 1]]]
        ###################################################
        return beta

    def sequence_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - prob: A float number of P(x_1:x_T | λ)
        """
        ###################################################
        alpha = self.forward(Osequence)
        prob = np.sum(alpha[:, -1])
        ###################################################
        return prob

    def posterior_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - prob: (num_state*L) A numpy array of P(s_t = i|O, λ)
        """
        S = len(self.pi)
        L = len(Osequence)
        ###################################################
        alpha = self.forward(Osequence)
        beta = self.backward(Osequence)
        seq_prob = self.sequence_prob(Osequence)
        prob = alpha * beta / seq_prob
        ###################################################
        return prob

    #TODO:
    def likelihood_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - prob: (num_state*num_state*(L-1)) A numpy array of P(X_t = i, X_t+1 = j | O, λ)
        """
        S = len(self.pi)
        L = len(Osequence)
        prob = np.zeros([S, S, L - 1])
        ###################################################
        alpha = self.forward(Osequence)
        beta = self.backward(Osequence)
        seq_prob = self.sequence_prob(Osequence)
        for t in range(L - 1):
            for i in range(S):
                for j in range(S):
                    prob[i, j, t] = alpha[i, t] * self.A[i, j] * \
                                     self.B[j, self.obs_dict[Osequence[t+1]]] * beta[j, t+1]
        prob /= seq_prob
        ###################################################
        return prob

    def viterbi(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - path: A List of the most likely hidden state path k* (return state instead of idx)
        """
        path = []
        ###################################################
        # Q3.3 Edit here
        # initialize
        path_idx = []
        S = len(self.pi)
        L = len(Osequence)
        cur_prob = np.zeros((S, L))
        max_idx = np.zeros_like(cur_prob, dtype=int)
        cur_prob[:, 0] = self.pi * self.B[:, self.obs_dict[Osequence[0]]]

        for t in range(1, L):
            temp = self.A * np.array(cur_prob[:, t - 1])[:, None]
            max_idx[:, t] = np.argmax(temp, axis=0)
            cur_prob[:, t] = self.B[:, self.obs_dict[Osequence[t]]] * np.max(temp, axis=0)
        path_idx.append(np.argmax(cur_prob[:, -1]))

        for t in range(L - 1, 0, -1):
            path_idx.append(max_idx[path_idx[-1], t])

        for idx in path_idx[::-1]:
            for state, cur_idx in self.state_dict.items():
                if cur_idx == idx:
                    path.append(state)
        ###################################################
        return path

















