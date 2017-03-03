from abc import ABCMeta, abstractmethod, abstractclassmethod
import copy
import numpy as np
import heapq
import utils


class Potential(metaclass=ABCMeta):

    def __init__(self, log_c=0):
        self.log_c = log_c

    def __lt__(self, other):
        return self.log_c < other.log_c

    def __gt__(self, other):
        return self.log_c > other.log_c

    @abstractmethod
    def __mul__(self, other):
        pass

    @abstractclassmethod
    def from_observation(cls, obs, args):
        pass

    @abstractclassmethod
    def default(cls, args):
        pass

    def copy(self):
        return copy.deepcopy(self)

    @abstractmethod
    def rand(self):
        pass

    @abstractmethod
    def mean(self):
        pass


class Message:

    def __init__(self, max_k=100):
        self.potentials = []  # potentials
        self.h = []           # heap for fast pruning
        self.max_k = max_k    # max capacity

    def __mul__(self, other):
        message = Message()
        for p1 in self.potentials:
            for p2 in other.potentials:
                message.potentials.append(p1 * p2)
        return message

    def __len__(self):
        return len(self.potentials)

    def add_potential(self, p):
        k = len(self.potentials)
        if k == self.max_k:
            k = heapq.heappop(self.h)[1]
            self.potentials[k] = p
        else:
            self.potentials.append(p)
        if k > 0:
            # push no-change messages to heap
            # skip for k = 0
            heapq.heappush(self.h, (p.log_c, k))

    # P(potential)
    def pp(self):
        return utils.normalize_exp(self.log_c())

    # first k potentials belong to change probabilities
    def cpp(self, k=1):
        return np.sum(self.pp()[:k])

    def log_likelihood(self):
        return utils.log_sum_exp(self.log_c())

    def log_c(self):
        return np.asarray([p.log_c for p in self.potentials])

    def mean(self):
        m = np.asarray([p.mean() for p in self.potentials])
        return np.dot(m.transpose(), self.pp())

    def get_ss(self):
        ss = np.asarray([p.get_ss() for p in self.potentials])
        return np.dot(ss.transpose(), self.pp())


class Model(metaclass=ABCMeta):

    def __init__(self, p1, prior):
        self.p1 = None
        self.log_p1 = None
        self.log_p0 = None
        self.set_p1(p1)
        self.prior = prior

    def set_p1(self, p1):
        self.p1 = p1
        self.log_p1 = np.log(p1)
        self.log_p0 = np.log(1-p1)

    @abstractclassmethod
    def load(cls, filename):
        pass

    @abstractmethod
    def save(self, filename):
        pass

    def generate_data(self, t):
        s = np.random.binomial(1, self.p1, t)               # change points
        h = np.zeros((t, self.m + self.n))    # hidden states
        v = np.zeros((t, self.m + self.n))    # observations
        for i in range(t):
            if i == 0 or s[i] == 1:
                # generate random state:
                h[i, :] = self.prior.rand()
            else:
                # copy previous state
                h[i, :] = h[i-1, :]
            # generate observation
            v[i, :] = self.rand_obs(h[i, :])
        return Data(s, h, v)

    @abstractmethod
    def rand_obs(self, state):
        pass

    def predict(self, alpha):
        m = Message()
        # add change component
        m.add_potential(Potential(self.prior.alpha, self.prior.a, self.prior.b, self.log_p1 + alpha.log_likelihood()))
        # add no-change components
        for p in alpha.potentials:
            m.add_potential(Potential(p.alpha, p.a, p.b, p.log_c + self.log_p0))
        return m

    def update(self, predict, obs):
        m = Message()
        p_obs = Potential.from_observation(obs)
        for p in predict.potentials:
            m.add_potential(p * p_obs)
        return m

    def forward(self, obs):
        alpha = []
        alpha_predict = []
        for i in range(obs.shape[0]):
            if i == 0:
                m = Message()
                m.add_potential(Potential(self.prior.alpha, self.prior.a, self.prior.b, self.log_p1))
                m.add_potential(Potential(self.prior.alpha, self.prior.a, self.prior.b, self.log_p0))
                alpha_predict.append(m)
            else:
                alpha_predict.append(self.predict(alpha[-1]))
            alpha.append(self.update(alpha_predict[-1], obs[i, :]))
        return [alpha_predict, alpha]

    def backward(self, obs, start=0, length=0):
        if length == 0:
            length = obs.shape[0]
            start = length-1
        beta = []
        for i in range(start, start - length, -1):
            message = Message()
            # change
            p_obs = Potential.from_observation(obs[i, :], self.m, self.n)
            pot_change = p_obs.copy()
            if len(beta) > 0:
                temp = Message()
                for p in beta[-1].potentials:
                    temp.add_potential(p * self.prior)
                pot_change.log_c += self.log_p1 + temp.log_likelihood()
            message.add_potential(pot_change)
            # no change
            if len(beta) > 0:
                for p in beta[-1].potentials:
                    p2 = p * p_obs
                    p2.log_c += self.log_p0
                    message.add_potential(p2)
            beta.append(message)
        beta.reverse()
        return beta

    def filter(self, obs):
        alpha = self.forward(obs)[1]
        # compile result
        result = Result()
        result.cpp = [message.cpp() for message in alpha]
        result.mean = [message.mean() for message in alpha]
        result.ll = [alpha[-1].log_likelihood()]
        return result

    def smooth(self, obs):
        [alpha_predict, alpha] = self.forward(obs)
        beta = self.backward(obs)
        # compile result
        result = Result()
        for i in range(len(alpha)):
            gamma = alpha_predict[i] * beta[i]
            result.cpp.append(gamma.cpp(len(beta[i].potentials)))
            result.mean.append(gamma.mean())
        result.ll = [alpha[-1].log_likelihood()]
        return result

    def online_smooth(self, obs, lag):
        if lag == 0:
            return self.filter(obs)

        t = obs.shape[0]
        if lag >= t:
            return self.smooth(obs)

        result = Result()
        [alpha_predict, alpha] = self.forward(obs)
        beta = []

        # Run Fixed-Lag for alpha[0:T - lag]
        for i in range(t - lag + 1):
            beta = self.backward(obs, i + lag - 1, lag)
            gamma = alpha_predict[i] * beta[0]
            result.cpp.append(gamma.cpp(len(beta[0])))
            result.mean.append(gamma.mean())

        # Smooth alpha[T-lag+1:T] with last beta.
        for i in range(1, lag):
            gamma = alpha_predict[t - lag + i] * beta[i]
            result.cpp.append(gamma.cpp(len(beta[i])))
            result.mean.append(gamma.mean())

        result.ll = [alpha[-1].log_likelihood()]
        return result
