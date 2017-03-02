

def fit_gamma(x, scale=0):
    return fit_gamma_from_ss(np.asarray([np.mean(x), np.mean(np.log(x))]), scale)


def fit_gamma_from_ss(ss, scale=0):
    if scale == 0:
        # fit Gamma shape and scale
        mean_x = ss[0]
        mean_log_x = ss[1]
        log_mean_x = np.log(mean_x)
        shape = 0.5 / (log_mean_x - mean_log_x)
        for i in range(5):
            temp = mean_log_x - log_mean_x + np.log(shape) - psi(shape)
            temp /= shape * shape * (1 / shape - sp.polygamma(1, shape))
            shape = 1 / (1 / shape + temp)
        scale = mean_x / shape
    else:
        # fit Gamma shape only
        shape = inv_psi(ss[1] - np.log(scale))
    return np.asarray([shape, scale])


def fit_dirichlet(x, precision=0):
    return fit_dirichlet_from_ss(np.mean(np.log(x), axis=0), precision)


def fit_dirichlet_from_ss(ss, precision=0):
    if precision == 0:
        # fit Dirichlet mean and precision
        alpha = normalize(ss)
        for i in range(1000):
            alpha_new = inv_psi(ss + psi(np.sum(alpha)))
            if i > 10 and np.allclose(alpha, alpha_new):
                break
            alpha = alpha_new
        return alpha
    else:
        # fit Dirichlet mean only
        m = normalize_exp(ss)
        for i in range(1000):
            m_new = normalize(inv_psi(ss - np.dot(m, ss - psi(m * precision))))
            if i > 10 and np.isclose(m, m_new):
                break
            m = m_new
        return m * precision


def test_dirichlet_fit(self):
    alpha = np.asarray([1, 2, 3, 4, 5])
    alpha_est = np.zeros(alpha.shape)
    n = 100
    for i in range(n):
        x = np.random.dirichlet(alpha, 1000)
        alpha_est += utils.fit_dirichlet(x)
    np.testing.assert_array_less(np.abs((alpha_est / n) - alpha), 0.1*np.ones(alpha.shape))


def test_gamma_fit(self):
    [shape, scale] = [10, 1]
    [shape_est, scale_est] = [0, 0]
    n = 100
    for i in range(n):
        x = np.random.gamma(shape, scale, 1000)
        [shape_est, scale_est] = [shape_est, scale_est] + utils.fit_gamma(x)
    self.assertLess(abs(shape - shape_est/n), 0.1)
    self.assertLess(abs(scale - scale_est/n), 0.1)


def fit(self, obs, max_iter=100):
    min_iter = 20
    ll = []
    for i in range(max_iter):
        # part 1: forward backward
        [alpha_predict, alpha] = self.forward(obs)
        beta = self.backward(obs)

        # part 2: E-Step
        cpp = []
        ss = []
        for t in range(obs.shape[0]):
            gamma = alpha_predict[t] * beta[t]
            cpp.append(gamma.cpp())
            ss.append(gamma.get_ss())
        sum_cpp = np.sum(cpp)
        ss_w = np.dot(np.asarray(ss).transpose(), np.asarray(cpp)) / sum_cpp
        ss_p1 = sum_cpp / obs.shape[0]

        # part 3: M-Step
        self.prior.fit(ss_w)
        self.set_p1(ss_p1)

        # part 4: Evaluate
        ll.append(alpha[-1].log_likelihood())
        print(ll[-1])

        # part 5: Stop if converged
        if i > 0:
            ll_diff = ll[i] - ll[i-1]
            if ll_diff < 0:
                print('Likelihood decreased!')
                break
            if i > min_iter and ll_diff < 1e-6:
                print('converged!')
                break

    result = self.smooth(obs)
    result.ll = ll
    return result


def fit_experiment(work_dir):

    results_dir = work_dir + '/py'
    utils.find_or_create(results_dir)

    data = Data.load(work_dir + '/data')
    real_model = Model.load(work_dir + '/model.txt')

    # Generate random model
    alpha = np.random.dirichlet(np.ones(real_model.m)) * np.random.rand() * 10
    a = np.random.rand(real_model.n) * 10
    b = np.ones(real_model.n)
    p1 = 0.0001
    model = Model(p1, alpha, a, b)

    # Fit model
    result = model.fit(data.v)
    result.save(results_dir + '/fit')