__author__ = "Nikolay Bryskin" # https://github.com/nikicat

import numpy as np
import sys
from collections import deque
from scipy.linalg import get_blas_funcs

gemm = get_blas_funcs('gemm')

#np.dot = lambda a, b: gemm(1, a, b)

OFFSET = 3000
SKIP_SIZE = 27
N_CHANNELS = 21

try:
    profile(lambda: 1)
except NameError:
    def profile(func):
        return func


class DelayedRLSPredictor:
    def __init__(self, n_channels, target_channel, M=3, lambda_=0.999, delta=100, delay=0, mu=0.3):
        self._target_channel = target_channel
        self._M = M
        self._lambda = lambda_
        self._delay = delay
        self._mu = mu
        size = M * n_channels
        self._w = np.zeros((size,))
        self._P = delta * np.eye(size)
        self._delta = delta
        self.regressors = deque(maxlen=M + delay + 1)

    @profile
    def predict(self, sample):
        self.regressors.append(sample)
        regressors = np.array(self.regressors)
        if regressors.shape[0] > self._delay + self._M:
            # predicted var x(t)
            predicted = regressors[-1, self._target_channel]

            # predictor var [x(t - M), x(t - M + 1), ..., x(t - delay)]
            predictor = regressors[- self._M - self._delay - 1: - self._delay - 1].flatten()  #

            # update helpers
            pi = np.dot(predictor, self._P)
            #print('pi.shape', pi.shape)
            #print('pi.flags', pi.flags)
            k = pi / (self._lambda + np.dot(pi, predictor))
            #print('k.shape', k[:, None].shape)
            #print('k.flags', k[:, None].flags)
            k_ = k[:, None]
            pi_ = pi[None, :]
            dot_ = np.dot(k_, pi_)
            P_dot = self._P - dot_
            self._P = 1 / self._lambda * P_dot

            # update weights
            dw = (predicted - np.dot(self._w, predictor)) * k
            self._w = self._w + self._mu * dw

            # return prediction x(t + delay)
            prediction = np.dot(self._w, regressors[- self._M:].flatten())
        else:
            # if lenght of regressor less than M + delay + 1 return 0
            prediction = 0

        return prediction


if __name__ == '__main__':
    experiment_id = input()
    channels = [15]
    rls = DelayedRLSPredictor(
        n_channels=len(channels), target_channel=channels.index(15), M=344,
        lambda_=0.999999, delta=160, delay=SKIP_SIZE, mu=1
    )

    for i in range(OFFSET):
        cur_data = np.array([float(j) for j in input().split()])
        prediction = rls.predict(cur_data[channels])

    print(prediction)
    sys.stdout.flush()

    while True:
        cur_data = np.array([float(i) for i in input().split()])
        prediction = rls.predict(cur_data[channels])
        print(prediction)
        sys.stdout.flush()
