#
# very simple HMM functions
#
# adapted from and therefore strongly basing on : 
#    https://github.com/AdeveloperdiAry/HiddenMArkovModel/tree/mAster/pArt4

import numpy as np

np.seterr(invalid='ignore')

def new_model(num_states=10, num_symbols=4):
    # alternate starting condition (equal prob.)
    pi = np.ones(num_states)
    pi = pi / np.sum(pi)

    A = np.random.rand(num_states,num_states)
    A = A / np.sum(A, axis=1)

    B = np.random.rand(num_states,num_symbols) 
    B = B / np.sum(B, axis=1).reshape((-1, 1))

    return(A,B,pi)


def forward(O, A, B, pi):
    alpha = np.zeros((O.shape[0], A.shape[0]))
    alpha[0, :] = pi * B[:, O[0]]

    for t in range(1, O.shape[0]):
        for j in range(A.shape[0]):
            alpha[t, j] = alpha[t - 1].dot(A[:, j]) * B[j, O[t]]

    return alpha


def backward(O, A, B):
    beta = np.zeros((O.shape[0], A.shape[0]))

    # setting beta(T) = 1
    beta[O.shape[0] - 1] = np.ones((A.shape[0]))

    # loop in backward way from T-1 to 1
    # due to python indexing the actual loop will be T-2 to 0
    for t in range(O.shape[0] - 2, -1, -1):
        for j in range(A.shape[0]):
            beta[t, j] = (beta[t + 1] * B[:, O[t + 1]]).dot(A[j, :])

    return beta


def baum_welch(O, AA, BB, pi, n_iter=100):
    M = AA.shape[0]
    T = len(O)
    A = np.array(AA)
    B = np.array(BB)

    for n in range(n_iter):
        alpha = forward(O, A, B, pi)
        beta = backward(O, A, B)

        xi = np.zeros((M, M, T - 1))
        for t in range(T - 1):
            denominator = np.dot(np.dot(alpha[t, :].T, A) * B[:, O[t + 1]].T, beta[t + 1, :])
            for i in range(M):
                numerator = alpha[t, i] * A[i, :] * B[:, O[t + 1]].T * beta[t + 1, :].T
                xi[i, :, t] = numerator / denominator

        gamma = np.sum(xi, axis=1)
        A = np.sum(xi, 2) / np.sum(gamma, axis=1).reshape((-1, 1))
        A[np.isnan(A)] = 0

        # add additional T'th element in gamma
        gamma = np.hstack((gamma, np.sum(xi[:, :, T - 2], axis=0).reshape((-1, 1))))

        K = B.shape[1]
        denominator = np.sum(gamma, axis=1)
        for l in range(K):
            B[:, l] = np.sum(gamma[:, O == l], axis=1)

        B = np.divide(B, denominator.reshape((-1, 1)))

    return (A, B)


def viterbi(O, A, B, pi):
    T = O.shape[0]
    M = A.shape[0]

    delta = np.zeros((T, M))
    delta[0, :] = pi * B[:, O[0]]

    psi = np.zeros((T - 1, M))

    for t in range(1, T):
        for j in range(M):
            # same as forward probability
            probability = delta[t - 1] * (A[:, j]) * B[j, O[t]]

            # this is our most probable state given previous state at time t (1)
            psi[t - 1, j] = np.argmax(probability)

            # this is the probability of the most probable state (2)
            delta[t, j] = np.max(probability)

    # path array
    q= np.zeros(T)

    # find the most probable last hidden state
    last_state = np.argmax(delta[T - 1, :])

    q[0] = last_state

    backtrack_index = 1
    for i in range(T - 2, -1, -1):
        q[backtrack_index] = psi[i, int(last_state)]
        last_state = psi[i, int(last_state)]
        backtrack_index += 1

    # flip the path array since we were backtracking
    q= np.flip(q, axis=0)

    return q.astype(int)



