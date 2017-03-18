/*
 * Author: Baris Kurt
 * E-mail: bariskurt@gmail.com
 *
 */

#include <pml/pml.hpp>

#include <queue>

using namespace pml;

#define MAX_COMPONENTS 100

namespace old {
  class Potential {

    public:
      Potential(const Vector &alpha_, const Vector &a_,
                const Vector &b_, double log_c_ = 0)
          : alpha(alpha_), a(a_), b(b_), log_c(log_c_) {}

      Potential(size_t M, size_t N) {
        if (M > 0)
          alpha = Vector::ones(M);
        if (N > 0) {
          a = Vector::ones(N) * 10;
          b = Vector::ones(N);
        }
        log_c = 0;
      }

      static Potential
      fromObservation(const Vector &obs, size_t m, size_t n) {
        Potential p(m, n);
        if (m > 0) {
          Vector obs_alpha = obs.getSlice(0, m);
          p.alpha = obs_alpha + 1;
          p.log_c = gammaln(sum(obs_alpha) + 1) - gammaln(sum(obs_alpha) + m);
        }
        if (n > 0) {
          p.a = obs.getSlice(m, m + n) + 1;
          p.b = Vector::ones(n);
        }
        return p;
      }

      friend Potential operator*(const Potential &p1, const Potential &p2) {
        Potential p = p1;
        p.log_c += p2.log_c;
        if (p.alpha.size() > 0) {
          p.alpha += p2.alpha - 1;
          p.log_c += gammaln(sum(p1.alpha)) - sum(gammaln(p1.alpha)) +
                     gammaln(sum(p2.alpha)) - sum(gammaln(p2.alpha)) +
                     sum(gammaln(p1.alpha + p2.alpha - 1)) -
                     gammaln(sum(p1.alpha + p2.alpha - 1));
        }
        if (p.a.size() > 0) {
          p.a += p2.a - 1;
          p.b = (p1.b * p2.b) / (p1.b + p2.b);
          p.log_c += sum(gammaln(p.a)) + sum(p.a * log(p.b))
                     - sum(gammaln(p1.a)) - sum(p1.a * log(p1.b))
                     - sum(gammaln(p2.a)) - sum(p2.a * log(p2.b));
        }
        return p;
      }

    public:
      size_t size() const {
        return alpha.size() + a.size();
      }

    public:
      Vector rand() const {
        Vector state;
        if (alpha.size() > 0)
          state = Dirichlet(alpha).rand();
        for (size_t i = 0; i < a.size(); ++i)
          state.append(Gamma(a[i], b[i]).rand(1));
        return state;
      }

      Vector mean() const {
        Vector mean;
        if (alpha.size() > 0)
          mean = normalize(alpha);
        if (a.size() > 0)
          mean.append(a * b);
        return mean;
      }

      Vector get_ss() const {
        Vector ss;
        if (alpha.size() > 0)
          ss = psi(alpha) - psi(sum(alpha));
        if (a.size() > 0) {
          ss.append(a * b);
          ss.append(psi(a) + log(b));
        }
        return ss;
      }

      void fit(Vector ss) {
        size_t m = alpha.size();
        size_t n = a.size();
        if (m > 0)
          alpha = Dirichlet::fit(ss.getSlice(0, m), m).alpha;
        for (size_t i = 0; i < n; ++i) {
          Gamma g = Gamma::fit(ss[m + i], ss[m + n + i], 1);
          a[i] = g.a;
          b[i] = g.b;
        }
      }

    public:
      bool operator<(const Potential &other) const {
        return this->log_c < other.log_c;
      }

    public:
      Vector alpha;
      Vector a;
      Vector b;
      double log_c;
    };


    class Message {

      private:
        struct HeapItem {
          double log_c;
          size_t idx;

          bool operator<(const HeapItem &other) const {
            return log_c < other.log_c;
          }

          bool operator>(const HeapItem &other) const {
            return log_c > other.log_c;
          }
        };

      public:
        Message(size_t max_components_ = 0) : max_components(max_components_) {}

        friend Message operator*(const Message &m1, const Message &m2) {
          Message m(0);
          for (const Potential &p1 : m1.potentials)
            for (const Potential &p2 : m2.potentials)
              m.add_potential(p1 * p2);
          return m;
        }

        void add_potential(const Potential &p) {
          if (potentials.empty() || max_components == 0) {
            potentials.emplace_back(p);
            log_c.append(p.log_c);
          } else {
            add_prune_potential(p);
          }
        }

        void add_prune_potential(const Potential &p) {
          size_t k = potentials.size();
          if (k == max_components) {
            k = heap.top().idx;
            heap.pop();
            potentials[k] = p;
            log_c[k] = p.log_c;
          } else {
            potentials.emplace_back(p);
            log_c.append(p.log_c);
          }
          heap.push({p.log_c, k});
        }

        size_t size() const {
          return potentials.size();
        }

        double cpp(size_t k = 1) const {
          Vector pp = normalizeExp(log_c);
          double result = 0;
          for (size_t i = 0; i < k; ++i)
            result += pp[i];
          return result;
        }

        double log_likelihood() const {
          return logSumExp(log_c);
        }

        Vector mean() const {
          Matrix m;
          for (const Potential &p : potentials)
            m.appendColumn(p.mean());
          return dot(m, normalizeExp(log_c));
        }

        Vector get_ss() const {
          Matrix ss;
          for (const Potential &p : potentials)
            ss.appendColumn(p.get_ss());
          return dot(ss, normalizeExp(log_c));
        }

    public:
        size_t max_components;
        Vector log_c;
        std::vector<Potential> potentials;
        std::priority_queue<HeapItem,
            std::vector<HeapItem>, std::greater<HeapItem>> heap;
    };

    using MessageVector = std::vector<Message>;

    struct Data {

      Data() {}

      Data(const std::string &dir) {
        obs = Matrix::loadTxt(path_join({dir, "obs.txt"}));
        states = Matrix::loadTxt(path_join({dir, "states.txt"}));
        cps = Vector::loadTxt(path_join({dir, "cps.txt"}));
      }

      void saveTxt(const std::string &dir) {
        find_or_create(dir);
        obs.saveTxt(path_join({dir, "obs.txt"}));
        states.saveTxt(path_join({dir, "states.txt"}));
        cps.saveTxt(path_join({dir, "cps.txt"}));
      }

      Matrix obs;
      Matrix states;
      Vector cps;
    };

    class Result {

      public:
        void saveTxt(const std::string &dir) {
          find_or_create(dir);
          mean.saveTxt(path_join({dir, "mean.txt"}));
          cpp.saveTxt(path_join({dir, "cpp.txt"}));
          ll.saveTxt(path_join({dir, "ll.txt"}));
          score.saveTxt(path_join({dir, "score.txt"}));
        }

        void append(const Vector &s) {
          score.appendRow(s);
        }

      public:
        Matrix mean;
        Vector cpp;
        Vector ll;
        Matrix score; //  [precision , recall, f_score] matrix
    };

    class Model {

      public:
        Model(double p1_, size_t m_, size_t n_) : prior(m_, n_), m(m_), n(n_) {
          set_p1(p1_);
        }

        Model(double p1_, const Potential &p) : prior(p) {
          set_p1(p1_);
          m = p.alpha.size();
          n = p.a.size();
        }

        Model(const std::string &filename) : prior(0, 0) {
          Vector temp = Vector::loadTxt(filename);
          set_p1(temp[0]);
          m = temp[1];
          n = temp[2];
          if (m > 0)
            prior.alpha = temp.getSlice(3, 3 + m);
          if (n > 0) {
            prior.a = temp.getSlice(3 + m, 3 + m + n);
            prior.b = temp.getSlice(3 + m + n, 3 + m + 2 * n);
          }
        }

        void saveTxt(const std::string &filename) const {
          Vector header = {p1, (double) m, (double) n};
          Vector result = cat({header, prior.alpha, prior.a, prior.b});
          result.saveTxt(filename);
        }

        void set_p1(double p1_new) {
          p1 = p1_new;
          log_p1 = std::log(p1);
          log_p0 = std::log(1 - p1);
        }

        Vector rand(const Vector &state) const {
          Vector result;
          if (m > 0)
            result = Multinomial(state.getSlice(0, m), 100).rand();
          for (size_t i = 0; i < n; ++i)
            result.append(Poisson(state[m + i]).rand());
          return result;
        }

        Data generateData(size_t length) {
          Data data;
          Vector state = prior.rand();
          Bernoulli bernoulli(p1);
          for (size_t t = 0; t < length; t++) {
            int change = 0;
            if (t > 0 && bernoulli.rand()) {
              state = prior.rand();
              change = 1;
            }
            data.states.appendColumn(state);
            data.obs.appendColumn(rand(state));
            data.cps.append(change);
          }
          return data;
        }

        Message predict(const Message &msg) {
          Message msg_next(MAX_COMPONENTS);
          // add change component
          Potential p_change = prior;
          p_change.log_c = log_p1 + msg.log_likelihood();
          msg_next.add_potential(p_change);
          // add no-change components
          for (const Potential &p : msg.potentials)
            msg_next.add_potential(
                Potential(p.alpha, p.a, p.b, p.log_c + log_p0));
          return msg_next;
        }

        Message update(const Message &msg, const Vector &obs) {
          Message msg_next(MAX_COMPONENTS);
          Potential p_obs = Potential::fromObservation(obs, m, n);
          for (const Potential &p : msg.potentials)
            msg_next.add_potential(p * p_obs);
          return msg_next;
        }

        void forward(const Matrix &obs) {
          alpha.clear();
          alpha_predict.clear();
          for (size_t i = 0; i < obs.ncols(); i++) {
            if (i == 0) {
              alpha_predict.emplace_back(MAX_COMPONENTS);
              alpha_predict.back().add_potential(
                  Potential(prior.alpha, prior.a, prior.b, log_p1));
              alpha_predict.back().add_potential(
                  Potential(prior.alpha, prior.a, prior.b, log_p0));
            } else {
              alpha_predict.push_back(predict(alpha.back()));
            }
            alpha.push_back(update(alpha_predict.back(), obs.getColumn(i)));
          }
        }

        void backward(const Matrix &obs, size_t start = 0, size_t steps = 0) {
          if (steps == 0) {
            steps = obs.ncols();
            start = obs.ncols() - 1;
          }
          beta.clear();
          for (size_t i = 0; i < steps; ++i) {
            Message msg(MAX_COMPONENTS);
            Message temp(MAX_COMPONENTS);
            auto p_obs = Potential::fromObservation(obs.getColumn(start - i), m, n);
            // change
            if (!beta.empty()) {
              for (const Potential &p : beta.back().potentials)
                temp.add_potential(p_obs * p);
              p_obs.log_c += log_p1 + temp.log_likelihood();
            }
            msg.add_potential(p_obs);
            // no change
            if (!beta.empty()) {
              for (Potential &p : temp.potentials) {
                p.log_c += log_p0;
                msg.add_potential(p);
              }
            }
            beta.push_back(msg);
          }
          std::reverse(beta.begin(), beta.end());
        }

        Result filtering(const Matrix &obs) {
          // Run forward
          forward(obs);

          // Calculate mean and cpp
          Result result;
          for (Message &message : alpha) {
            result.mean.appendColumn(message.mean());
            result.cpp.append(message.cpp());
          }
          result.ll.append(alpha.back().log_likelihood());
          return result;
        }

        Result smoothing(const Matrix &obs) {
          // Run Forward - Backward
          forward(obs);
          backward(obs);

          // Calculate Smoothed density
          Result result;
          for (size_t i = 0; i < obs.ncols(); ++i) {
            Message gamma = alpha_predict[i] * beta[i];
            result.mean.appendColumn(gamma.mean());
            result.cpp.append(gamma.cpp(beta[i].size()));
          }
          result.ll.append(alpha.back().log_likelihood());
          return result;
        }

        Result online_smoothing(const Matrix &obs, size_t lag) {
          if (lag == 0)
            return filtering(obs);

          if (lag >= obs.ncols())
            return smoothing(obs);

          Result result;

          // Go forward
          forward(obs);

          // Run Fixed-Lags for alpha[0:T-lag]
          for (size_t t = 0; t <= obs.ncols() - lag; ++t) {
            backward(obs, t + lag - 1, lag);
            Message gamma = alpha_predict[t] * beta.front();
            result.mean.appendColumn(gamma.mean());
            result.cpp.append(gamma.cpp(beta.front().size()));
          }
          // Smooth alpha[T-lag+1:T] with last beta.
          for (size_t i = 1; i < lag; ++i) {
            Message gamma = alpha_predict[obs.ncols() - lag + i] * beta[i];
            result.mean.appendColumn(gamma.mean());
            result.cpp.append(gamma.cpp(beta[i].size()));
          }
          result.ll.append(alpha.back().log_likelihood());
          return result;
        }

        Result learn_parameters(const Matrix &obs, size_t max_iter = 100) {

          size_t min_iter = 20;
          Vector ll;

          for (size_t iter = 0; iter < max_iter; ++iter) {

            // Forward_backward
            forward(obs);
            backward(obs);

            // Part 1: E-Step
            Vector cpp;
            Matrix ss;
            for (size_t i = 0; i < alpha.size(); ++i) {
              Message gamma = alpha_predict[i] * beta[i];
              cpp.append(gamma.cpp(beta[i].size()));
              ss.appendColumn(gamma.get_ss());
            }

            // Part 2: Evaluate
            ll.append(alpha.back().log_likelihood());
            std::cout << "iter: " << iter
                      << "\tlog-likelihood : " << ll.last() << std::endl;
            if (iter > 0) {
              double ll_diff = ll[iter] - ll[iter - 1];
              if (ll_diff < 0) {
                std::cout << "!!! log-likelihood decreased: " << -ll_diff
                          << "\n";
                break;
              }
              if (iter > min_iter && (ll_diff < 1e-3)) {
                std::cout << "converged.\n";
                break;
              }
            }

            // Part 3: M-Step
            prior.fit(dot(ss, normalize(cpp)));
            set_p1(mean(cpp));

          }
          Result result = smoothing(obs);
          result.ll = ll;
          return result;
        }

      public:
        double p1;
        double log_p0;
        double log_p1;
        Potential prior;
        size_t m;
        size_t n;

      private:
        std::vector<Message> alpha;
        std::vector<Message> alpha_predict;
        std::vector<Message> beta;
    };


    Vector evaluate(const Vector &cps, const Vector &cpp,
                    const double threshold, const size_t window) {

      Vector cps_true, cps_est;
      for (size_t i = 0; i < cpp.size(); ++i) {
        if (cps(i) > threshold)
          cps_true.append(i);
        if (cpp(i) > threshold)
          cps_est.append(i);
      }

      Vector true_points(cps_true.size());
      Vector pred_points(cps_est.size());
      for (size_t i = 0; i < cps_est.size(); ++i) {
        for (size_t j = 0; j < cps_true.size(); ++j) {
          double dist = cps_est(i) - cps_true(j);
          if (-1 <= dist && dist <= window) {
            true_points(j) = 1;
            pred_points(i) = 1;
          }
        }
      }
      double true_positives = sum(true_points);
      double false_positives = pred_points.size() - sum(pred_points);
      double false_negatives = true_points.size() - true_positives;
      double precision = true_positives / (true_positives + false_positives);
      double recall = true_positives / (true_positives + false_negatives);
      double f_score = 2 * (precision * recall) / (precision + recall);

      return Vector({precision, recall, f_score});
    }
} // namespace