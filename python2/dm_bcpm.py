import numpy as np
import bcpm
import utils


class DirichletPotential(bcpm.Potential):

    def __init__(self, alpha, log_p1=0):
        super(bcpm.Potential, self).__init__(log_p1)
        self.alpha = alpha

    def __mul__(self, other):
        p = self.copy()
        p.log_c += other.log_c
        p.alpha = self.alpha + other.alpha - 1
        p.log_c += utils.gammaln(np.sum(self.alpha)) - np.sum(utils.gammaln(self.alpha))
        p.log_c += utils.gammaln(np.sum(other.alpha)) - np.sum(utils.gammaln(other.alpha))
        p.log_c += np.sum(utils.gammaln(p.alpha)) - utils.gammaln(np.sum(p.alpha))
        return p

    def rand(self):
        return np.random.dirichlet(self.alpha)

    def mean(self):
        return utils.normalize(self.alpha)

    @classmethod
    def from_observations(cls, obs, args=()):
        sum_obs = np.sum(obs)
        log_c = utils.gammaln(sum_obs + 1) - utils.gammaln(sum_obs + len(obs))
        return DirichletPotential(obs+1, log_c)

    @classmethod
    def default(cls, args):
        return DirichletPotential(np.ones(args))

bcpm.Potential.

class DM_Model(bcpm.Model):

    def __init__(self, p1, prior):
        super(bcpm.Model, self).__init__(p1, prior)

    @classmethod
    def load(cls, filename):
        pass

    def save(self, filename):
        pass