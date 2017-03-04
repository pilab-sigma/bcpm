import bcpm
from bcpm import utils

import numpy as np
import unittest


class TestUtils(unittest.TestCase):

    def test_load_save(self):
        x = np.random.rand(4, 5)
        utils.save_txt('/tmp/x.txt', x)
        y = utils.load_txt('/tmp/x.txt')
        np.testing.assert_array_almost_equal(x, y)

    def test_gammaln(self):
        v = np.asarray([1, 2, 3, 4, 5])
        gammaln_v = np.asarray([0, 0, 0.693147, 1.791759, 3.178053])
        np.testing.assert_array_almost_equal(utils.gammaln(v), gammaln_v)

    def test_normalize(self):
        v = np.asarray([1, 2, 3, 4, 5])
        nv = np.asarray([0.066667, 0.133333, 0.2, 0.266667, 0.333333])
        np.testing.assert_array_almost_equal(utils.normalize(v), nv)

    def test_normalize_exp(self):
        v = np.asarray([1, 2, 3, 4, 5])
        log_v = np.log(v)
        np.testing.assert_array_almost_equal(utils.normalize(v), utils.normalize_exp(log_v))

    def test_log_sum_exp(self):
        v = np.asarray([1, 2, 3, 4, 5])
        np.testing.assert_array_almost_equal(utils.log_sum_exp(v), np.log(np.sum(np.exp(v))))

    def test_psi(self):
        v = np.asarray([1, 2, 3, 4, 5])
        psi_v = np.asarray([-0.57721566, 0.42278434, 0.92278434, 1.25611767, 1.50611767])
        np.testing.assert_array_almost_equal(utils.psi(v), psi_v)

    def test_inv_psi(self):
        for x in [1e-6, 0.1, 1, 2, 1000]:
            np.testing.assert_almost_equal(x, utils.inv_psi(utils.psi(x)))


# class TestDataMethods(unittest.TestCase):
#
#     def test_load_save(self):
#         t = 100
#         d = 5
#         data = Data(np.zeros(t), np.random.rand(t, d), np.random.rand(t, d))
#         data.save('/tmp/data/')
#         data2 = Data.load('/tmp/data/')
#         np.testing.assert_array_almost_equal(data.s, data2.s)
#         np.testing.assert_array_almost_equal(data.h, data2.h)
#         np.testing.assert_array_almost_equal(data.v, data2.v)
#
#
# class TestModelMethods(unittest.TestCase):
#
#     def test_load_save(self):
#         m = Model(0.01, [1, 1, 1, 1], [10, 10, 10], [1, 1, 1])
#         m.save('/tmp/model1.txt')
#         m2 = Model.load('/tmp/model1.txt')
#         self.assertEqual(m.p1, m2.p1)
#         np.testing.assert_array_almost_equal_nulp(m.prior.alpha, m2.prior.alpha)
#         np.testing.assert_array_almost_equal_nulp(m.prior.a, m2.prior.a)
#         np.testing.assert_array_almost_equal_nulp(m.prior.b, m2.prior.b)
#
#     def test_generate_data(self):
#         t = 100
#         # Generate Dirichlet only
#         m = Model(0.1, [1, 1, 1], [], [])
#         data = m.generate_data(t)
#         data.save('/tmp/data1')
#
#         # Generate Gamma only
#         m2 = Model(0.1, [], [10, 10], [1, 1])
#         data2 = m2.generate_data(t)
#         data2.save('/tmp/data2')
#
#         # Generate Coupled
#         m3 = Model(0.1, [1, 1, 1, 1], [10, 10], [1, 1])
#         data3 = m3.generate_data(t)
#         data3.save('/tmp/data3')
#
#
# class TestPotential(unittest.TestCase):
#
#     def test_deepcopy(self):
#         p1 = Potential.default(3, 2)
#         p2 = p1.copy()
#
#         p2.alpha = np.asarray([1, 2, 3])
#         p2.a = np.asarray([6, 7])
#         p2.b = np.asarray([2, 3])
#         p2.log_c = 2
#         self.assertNotEqual(p1.log_c, p2.log_c)
#
#     def test_comparison(self):
#         p1 = Potential.default(3, 2)
#         p2 = Potential.default(3, 2)
#         p2.log_c = 2
#         self.assertTrue(p1 < p2)
#         self.assertTrue(p2 > p1)
#
#     def test_multiplication(self):
#         # Part 1: Test Dirichlet multiplication
#         p1 = Potential([1, 2, 3, 4], [], [])
#         p2 = Potential([5, 6, 7, 8], [], [])
#         p3 = p1 * p2
#         np.testing.assert_almost_equal(p3.log_c, 2.62466487)
#         np.testing.assert_array_almost_equal(p3.alpha, np.asarray([5, 7, 9, 11]))
#
#         # Part 2: Test Gamma multiplication
#         p1 = Potential([], [10], [1])
#         p2 = Potential([], [5], [2])
#         p3 = p1 * p2
#         np.testing.assert_almost_equal(p3.log_c, -2.56996487)
#         np.testing.assert_almost_equal(p3.a, 14)
#         np.testing.assert_almost_equal(p3.b, 0.66666667)
#
#         # Part 2: Test Coupled multiplication
#         p1 = Potential([1, 2, 3, 4], [10], [1])
#         p2 = Potential([5, 6, 7, 8], [5], [2])
#         p3 = p1 * p2
#         np.testing.assert_almost_equal(p3.log_c, 2.62466487 - 2.56996487)
#         np.testing.assert_array_almost_equal(p3.alpha, np.asarray([5, 7, 9, 11]))
#         np.testing.assert_almost_equal(p3.a, 14)
#         np.testing.assert_almost_equal(p3.b, 0.66666667)
#
#     def test_from_observation(self):
#         obs = np.concatenate((utils.normalize([2, 3, 4, 5]), [7]))
#         p = Potential.from_observation(obs, 4, 1)
#         np.testing.assert_array_almost_equal(p.alpha, np.asarray([1.142857, 1.214286, 1.285714, 1.357143]))
#         np.testing.assert_almost_equal(p.log_c, -3.17805383)
#         np.testing.assert_almost_equal(p.a, 8)
#         np.testing.assert_almost_equal(p.b, 1)
#
#     def test_mean(self):
#         p = Potential([1, 2, 3, 4], [10], [1])
#         np.testing.assert_array_almost_equal(p.mean(), np.asarray([0.1, 0.2, 0.3, 0.4, 10]))
#
#     def test_ss(self):
#         p = Potential([1, 2, 3, 4], [10], [1])
#         ss = np.asarray([])
#         np.testing.assert_array_almost_equal(p.get_ss(), ss)
#
#     def test_rand_and_fit(self):
#         p = Potential([1, 2, 3, 4], [10, 5], [1, 2])
#         n = 1000
#         x = np.zeros((n, 6))
#         for i in range(n):
#             x[i, :] = p.rand()
#         ss1 = np.mean(np.log(x[:, 0:4]), axis=0)
#         ss2 = np.concatenate((np.mean(x[:, 4:6], axis=0), np.mean(np.log(x[:, 4:6]), axis=0)))
#         ss = np.concatenate((ss1, ss2))
#         p2 = Potential.default(4,2)
#         p2 = p.copy()
#         p2.fit(ss)

if __name__ == '__main__':
    unittest.main()