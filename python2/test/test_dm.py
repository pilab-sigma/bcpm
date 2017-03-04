import numpy as np
import unittest

from bcpm import bcpm_dm as dm
from bcpm import utils


class TestPotential(unittest.TestCase):

    def test_deepcopy(self):
        p1 = dm.DirichletPotential([1, 2, 3, 4], 0.5)
        p2 = p1.deepcopy()

        # alpha vectors are copied
        np.testing.assert_array_equal(p1.alpha, p2.alpha)
        self.assertIsNot(p1.alpha, p2.alpha)

        self.assertIs(p1.log_c, p2.log_c)
        p2.log_c = 1
        self.assertIsNot(p1.log_c, p2.log_c)
        self.assertNotEqual(p1.log_c, p2.log_c)

    def test_comparison(self):
        p1 = dm.DirichletPotential.default(5)
        p2 = dm.DirichletPotential.default(5)
        p2.log_c = 2
        self.assertTrue(p1 < p2)
        self.assertTrue(p2 > p1)

    def test_multiplication(self):
        p1 = dm.DirichletPotential([1, 2, 3, 4])
        p2 = dm.DirichletPotential([5, 6, 7, 8])
        p3 = p1 * p2
        np.testing.assert_almost_equal(p3.log_c, 2.62466487)
        np.testing.assert_array_almost_equal(p3.alpha, np.asarray([5, 7, 9, 11]))

    def test_from_observation(self):
        obs = utils.normalize([2, 3, 4, 5])
        p = dm.DirichletPotential.from_observation(obs)
        np.testing.assert_array_almost_equal(p.alpha, np.asarray([1.142857, 1.214286, 1.285714, 1.357143]))
        np.testing.assert_almost_equal(p.log_c, -3.17805383)

    def test_mean(self):
        p = dm.DirichletPotential([1, 2, 3, 4])
        np.testing.assert_array_almost_equal(p.mean(), np.asarray([0.1, 0.2, 0.3, 0.4]))