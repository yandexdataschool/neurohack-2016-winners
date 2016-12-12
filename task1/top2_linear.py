__author__ = "bendyna.ivan"

import numpy as np
import sys
from collections import deque
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings("ignore")

if sys.version_info.major == 2:
    input = raw_input

OFFSET = 3000
SKIP_SIZE = 28
TARGET_CHANNEL = 15
N_CHANNELS = 21

experiment_id = input()


class LinearPredictor:
    def __init__(self):
        self.limit = 6000
        self.window_size = 150
        self.X = np.zeros((self.limit, self.window_size))
        self.target = np.zeros(self.limit)
        self.count = 0
        self.second_channel = 1
        self.model = None
        self.period_learn = 100

    def predict(self, sample):
        c = self.count % self.limit
        c1 = (self.count + self.limit - 1) % self.limit
        self.X[c, 0] = sample[TARGET_CHANNEL]
        if self.count > 0:
            for i in xrange(1, self.window_size):
                self.X[c, i] = self.X[c1, i - 1]
        if self.count >= SKIP_SIZE:
            self.target[(self.count + self.limit - SKIP_SIZE) % self.limit] = sample[TARGET_CHANNEL]
        if self.count >= OFFSET and self.count % self.period_learn == 0:
            self.model = LinearRegression()
            temp = np.zeros((SKIP_SIZE, self.window_size))
            temp_target = np.zeros(SKIP_SIZE)
            for j in xrange(SKIP_SIZE):
                t = (self.count + self.limit - SKIP_SIZE + 1 + j) % self.limit
                temp[j] = self.X[t]
                temp_target[j] = self.target[t]
                self.X[t] = 0
                self.target[t] = 0
            self.model.fit(self.X, self.target)
            for j in xrange(SKIP_SIZE):
                t = (self.count + self.limit - SKIP_SIZE + 1 + j) % self.limit
                self.X[t] = temp[j]
                self.target[t] = temp_target[j]
        self.count += 1

        if self.count <= OFFSET:
            return 0
        else:
            return self.model.predict(self.X[c, :])[0]


rls = LinearPredictor()

for i in range(OFFSET):
    cur_data = list(map(float, input().split()))
    prediction = rls.predict(cur_data)

print(prediction)
sys.stdout.flush()

while True:
    cur_data = list(map(float, input().split()))
    prediction = rls.predict(cur_data)
    print(prediction)
    sys.stdout.flush()
