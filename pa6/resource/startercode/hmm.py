from __future__ import print_function
import json
import numpy as np


class HMM:

    def __init__(self, pi, A, B, obs_dict, state_dict):
        """
        - pi: (1*num_state) A numpy array of initial probabilities. pi[i] = P(Z_1 = s_i)
        - A: (num_state*num_state) A numpy array of transition probabilities. A[i, j] = P(Z_t = s_j|Z_{t-1} = s_i)
        - B: (num_state*num_obs_symbol) A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
        - obs_dict: A dictionary mapping each observation symbol to its index
        - state_dict: A dictionary mapping each state to its index
        """
        self.pi = pi
        self.A = A
        self.B = B
        self.obs_dict = obs_dict
        self.state_dict = state_dict

    def forward(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - alpha: (num_state*L) A numpy array where alpha[i, t-1] = P(Z_t = s_i, X_{1:t}=x_{1:t})
                 (note that this is alpha[i, t-1] instead of alpha[i, t])
        """
        S = len(self.pi)
        L = len(Osequence)
        O = self.find_item(Osequence)  # map to index
        alpha = np.zeros([S, L])
        ######################################################
        # TODO: compute and return the forward messages alpha
        ######################################################
        for i in range(S):
            k = O[0]
            m = self.obs_dict[Osequence[0]]
            alpha[i, 0] = self.pi[i] * self.B[i, k]

        for t in range(1, L):
            for s in range(S):
                sum = 0
                for s_prime in range(S):
                    sum += self.A[s_prime, s] * alpha[s_prime, t - 1]
                k = O[t]
                alpha[s, t] = self.B[s, k] * sum

        return alpha

    def backward(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - beta: (num_state*L) A numpy array where beta[i, t-1] = P(X_{t+1:T}=x_{t+1:T} | Z_t = s_i)
                    (note that this is beta[i, t-1] instead of beta[i, t])
        """
        S = len(self.pi)
        L = len(Osequence)
        O = self.find_item(Osequence)
        beta = np.zeros([S, L])
        #######################################################
        # TODO: compute and return the backward messages beta
        #######################################################
        for s in range(S):
            beta[s, L - 1] = 1

        for t in range(L - 2, -1, -1):
            for s in range(S):
                abbsum = 0
                for s_prime in range(S):
                    k = O[t + 1]
                    abbsum += self.A[s, s_prime] * self.B[s_prime, k] * beta[s_prime, t + 1]
                beta[s, t] = abbsum


        return beta

    def sequence_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - prob: A float number of P(X_{1:T}=x_{1:T})
        """

        #####################################################
        # TODO: compute and return prob = P(X_{1:T}=x_{1:T})
        #   using the forward/backward messages
        #####################################################
        L = len(Osequence)
        S = len(self.pi)

        alpha = self.forward(Osequence)
        beta = self.backward(Osequence)

        px_1t = 0.0
        for s in range(S):
            px_1t += alpha[s, L - 1] * beta[s, L - 1]

        return px_1t

    def posterior_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - gamma: (num_state*L) A numpy array where gamma[i, t-1] = P(Z_t = s_i | X_{1:T}=x_{1:T})
		           (note that this is gamma[i, t-1] instead of gamma[i, t])
        """
        ######################################################################
        # TODO: compute and return gamma using the forward/backward messages
        ######################################################################
        L = len(Osequence)
        S = len(self.pi)

        alpha = self.forward(Osequence)
        beta = self.backward(Osequence)
        px_1t = self.sequence_prob(Osequence)
        gamma = np.zeros([S, L])

        for t in range(L):
            for s in range(S):
                gamma[s, t] = alpha[s, t] * beta[s, t] / px_1t

        return gamma

    def likelihood_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - prob: (num_state*num_state*(L-1)) A numpy array where prob[i, j, t-1] =
                    P(Z_t = s_i, Z_{t+1} = s_j | X_{1:T}=x_{1:T})
        """
        S = len(self.pi)
        L = len(Osequence)
        prob = np.zeros([S, S, L - 1])
        #####################################################################
        # TODO: compute and return prob using the forward/backward messages
        #####################################################################
        O = self.find_item(Osequence)
        alpha = self.forward(Osequence)
        beta = self.backward(Osequence)
        px_1t = self.sequence_prob(Osequence)

        for t in range(L - 1):
            for s in range(S):
                for s_prime in range(S):
                    prob[s, s_prime, t] = alpha[s, t] * self.A[s, s_prime] * self.B[s_prime, O[t + 1]] * beta[
                        s_prime, t + 1]
        prob /= px_1t


        return prob

    def viterbi(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - path: A List of the most likely hidden states (return actual states instead of their indices;
                    you might find the given function self.find_key useful)
        """
        path = []
        ################################################################################
        # TODO: implement the Viterbi algorithm and return the most likely state path
        ################################################################################
        S = len(self.pi)
        L = len(Osequence)
        alpha = self.forward(Osequence)
        beta = self.backward(Osequence)
        O = self.find_item(Osequence)

        delta = np.zeros([S, L])
        Delta = np.zeros([S, L], dtype=int)

        path_ix = []

        for s in range(S):
            delta[s, 0] = self.pi[s] * self.B[s, O[0]]

        for t in range(1, L):
            for s in range(S):
                max_ad = -1
                for s_prime in range(S):
                    val = self.A[s_prime, s] * delta[s_prime, t - 1]
                    if val > max_ad:
                        max_ad = val
                        Delta[s, t] = s_prime
                delta[s, t] = self.B[s, O[t]] * max_ad

        zt = np.argmax(delta[:,-1])
        path_ix.append(zt)
        i = 0

        for t in range(L - 1, 0, -1):
            zt = path_ix[i]

            zt_1 = Delta[zt, t]
            path_ix.append(zt_1)
            i += 1

        for i in reversed(path_ix):
            for state, ix in self.state_dict.items():
                if ix == i:
                    path.append(state)



        return path

    # DO NOT MODIFY CODE BELOW
    def find_key(self, obs_dict, idx):
        for item in obs_dict:
            if obs_dict[item] == idx:
                return item

    def find_item(self, Osequence):
        O = []
        for item in Osequence:
            O.append(self.obs_dict[item])
        return O
