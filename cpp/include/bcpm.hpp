#ifndef BCPM_H_
#define BCPM_H_

#include <pml/pml.hpp>

using namespace pml;


/*
 * Change Point Data Structure
 *
 * There are 3 data structures for the change point process
 *
 *  cps    : change points binary vector denoting change positions
 *  states : D x N hidden state matrix.
 *  obs    : K x N observation matrix.
 *
 *  N = number of epochs
 *  D = dimension of the state vector
 *  K = dimension of the observation vector
 *
 * */
class ChangePointData {

  public:
    // Saves the data matrices to the folder "dirname"
    // The fodler is created if it does not exist
    void saveTxt(const std::string &dirname){
      find_or_create(dirname);
      obs.saveTxt(make_path({dirname, "obs.txt"}));
      states.saveTxt(make_path({dirname, "states.txt"}));
      cps.saveTxt(make_path({dirname, "cps.txt"}));
    }

    // Loads the data matrices from folder "dirname"
    ChangePointData(const std::string &dirname){
      obs = Matrix::loadTxt(make_path({dirname, "obs.txt"}));
      states = Matrix::loadTxt(make_path({dirname, "states.txt"}));
      cps = Vector::loadTxt(make_path({dirname, "cps.txt"}));
    }

  public:
    Matrix obs;
    Matrix states;
    Vector cps;
};


// ----------- POTENTIAL ----------- //

class Potential {

  public:
    explicit Potential(double log_c_) : log_c(log_c_) {}

    virtual ~Potential() {}

  public:
    bool operator<(const Potential &that) const{
      return log_c < that.log_c;
    }

    bool operator>(const Potential &that) const{
      return log_c > that.log_c;
    }

  public:
    // Multiplication of two potentials
    virtual void operator*=(const Potential &p) = 0;
    virtual Potential* operator*(const Potential &p) const = 0;

  public:
    // Draw a random state vector from the potential
    virtual Vector rand() const = 0;

    // Returns the mean of the underlying distribution
    virtual Vector mean() const = 0;

    // Get sufficient statistics for the underlyting distribution
    virtual Vector get_ss() const = 0;

  public:
    double log_c;
};

// ----------- MESSAGE ----------- //

bool cmp_potentials(const Potential *p1, const Potential *p2) {
  return (*p1) > (*p2);
}

class Message {

  public:
    Message(size_t max_components_ = 0 ) : max_components(max_components_){}

    // Returns number of potentials
    size_t size() const {
      return potentials.size();
    }

    // Adds a potential with or without pruning
    void add_potential(Potential *p){
      potentials.push_back(p);
      // Prune if max_components is defined
      // First component is the change component, so the heap starts
      // from the second potential
      if( max_components > 0 ) {
        std::push_heap(++potentials.begin(), potentials.end(), cmp_potentials);
        if( potentials.size() > max_components ) {
          Potential *p = potentials.back();
          potentials.pop_back();
          delete p;
        }
      }
    }

    friend Message operator*(const Message &m1, const Message &m2){
      Message msg;
      for(Potential *p1 : m1.potentials)
        for(Potential *p2 : m2.potentials)
          msg.add_potential((*p1) * (*p2));
      return msg;
    }

    Vector get_log_c() const {
      Vector log_c(potentials.size());
      for(size_t i = 0; i < potentials.size(); ++i)
        log_c[i] = potentials[i]->log_c;
      return log_c;
    }

    // Relative probability of potentials
    Vector get_rpp() const {
      normalizeExp(get_log_c());
    }

    double log_likelihood() const {
      return logSumExp(get_log_c());
    }

  public:
    double get_cpp(size_t num_cpp = 1) const {
      Vector rpp = get_rpp();
      double result = 0;
      for(size_t i=0; i < num_cpp; ++i)
        result += rpp[i];
      return result;
    }

    Vector get_mean() const {
      Matrix params;
      for(Potential *p: potentials)
        params.appendColumn(p->mean());
      return dot(params, normalizeExp(get_consts()));
    }

  public:
    size_t max_components;
    std::vector<Potential*> potentials;

};


// ----------- MODEL ----------- //

class Model{

  public:
    explicit Model(double p1_){
      set_p1(p1_);
    }

    virtual ~Model(){
      delete prior;
    }

  public:
    void set_p1(double p1_new){
      p1 = p1_new;
      log_p1 = std::log(p1);
      log_p0 = std::log(1-p1);
    }

  public:

    // Generates a change point process of length "length" from the model.
    ChangePointData generateData(size_t length){
      ChangePointData data;
      Vector state = prior->rand();
      Bernoulli bernoulli(p1);
      for (size_t t=0; t<length; t++) {
        int change = 0;
        if (t > 0 && bernoulli.rand()) {
          state = prior->rand();
          change = 1;
        }
        data.states.appendColumn(state);
        data.obs.appendColumn(rand(state));
        data.cps.append(change);
      }
      return data;
    }

  public:
    virtual Potential* getChangePotential(double delta_log_c = 0) const = 0;
    virtual Potential* getNoChangePotential(double delta_log_c = 0) const = 0;
    virtual Potential* obs2Potential(const Vector &obs) const = 0;

    virtual Vector rand(const Vector &state) const = 0;
    virtual void fit(const Vector &ss, double p1_new) = 0;
    virtual void saveTxt(const std::string &filename) const = 0;
    virtual void loadTxt(const std::string &filename) = 0;
    virtual void print() const = 0;

  public:
    Potential *prior;
    double p1, log_p1, log_p0;
};


// ----------- RESULT ----------- //

class Result{

  public:
    void saveTxt(const std::string &dir){
      find_or_create(dir);
      mean.saveTxt(make_path({dir, "mean.txt"}));
      cpp.saveTxt(make_path({dir, "cpp.txt"}));
      ll.saveTxt(make_path({dir, "ll.txt"}));
      score.saveTxt(make_path({dir, "score.txt"}));
    }

    void append(const Vector &s){
      score.appendRow(s);
    }

  public:
    Matrix mean;
    Vector cpp;
    Vector ll;
    Matrix score; //  [precision , recall, f_score] matrix
};


// ----------- EVALUATOR ----------- //


class Evaluator{

  public:
    Evaluator(const Vector &cps_, double threshold_, size_t window_)
            : threshold(threshold_), window(window_){
      for(size_t i=0; i < cps_.size(); ++i){
        if( cps_(i) == 1)
          cps.append(i);
      }
    }

    Evaluator(const std::string &cps_file, double threshold_, size_t window_)
            : threshold(threshold_), window(window_){
      Vector cps_ = Vector::loadTxt(cps_file);
      for(size_t i=0; i < cps_.size(); ++i){
        if( cps_(i) == 1)
          cps.append(i);
      }
    }

  public:
    Vector evaluate(const std::string &cpp_file){
      return evaluate(Vector::loadTxt(cpp_file));
    }

    Vector evaluate(const Vector &cpp){
      // cpp to cps
      Vector cps_est;
      for(size_t i=0; i < cpp.size(); ++i){
        if( cpp(i) > threshold){
          cps_est.append(i);
        }
      }

      Vector true_points(cps.size());
      Vector pred_points(cps_est.size());
      for(size_t i=0; i < cps_est.size(); ++i){
        for(size_t j=0; j < cps.size(); ++j) {
          double dist = cps_est(i) - cps(j);
          if( -1 <= dist && dist <= window){
            true_points(j) = 1;
            pred_points(i) = 1;
          }
        }
      }

      double true_positives = sum(true_points);
      double false_positives = pred_points.size() - sum(pred_points);
      double false_negatives = true_points.size() - true_positives;

      // We know that our data definitely contains change points,
      // so, we set precision and recall to zero if true positives are zero.
      double precision = 0, recall = 0, f_score = 0;
      if ( true_positives > 0 ){
        precision = true_positives / (true_positives + false_positives);
        recall = true_positives / (true_positives + false_negatives);
        f_score = 2 * (precision*recall) / (precision + recall);
      }

      return Vector({precision, recall, f_score});
    }

  public:
    Vector cps; // binary vector storing change locations
    double threshold;
    size_t window;
};

// ----------- FORWARD-BACKWARD ----------- //

class ForwardBackward {

  public:
    ForwardBackward(Model *model_, int max_components_ = 100)
        :model(model_), max_components(max_components_){
      alpha.clear();
      alpha_predict.clear();
      beta.clear();
    }

  public:
    Message predict(const Message& prev){

      Message next(MAX_COMPONENTS);
      // add change component
      next.add_potential(model->getChangePotential(prev.log_likelihood()));
      // add no-change components
      for(const Potential *p : prev.potentials)
        msg_next.add_potential(Potential(p.alpha, p.a, p.b, p.log_c + log_p0));
      //return next;


      Message message(max_components);
      Vector consts;
      for(Potential *p : message.potentials){
        consts.append(p->log_c);
        p->log_c += model->log_p0;
      }
      message.add_potential(model->getChangePotential(logSumExp(consts)));
      // message.add_potential(model->prior, model->log_p1 + );
      return message;
    }

    Message update(const Message &prev, const Vector &obs){
      Message message = prev;
      Potential *p_obs = model->obs2Potential(obs);
      for(Potential *p : message.potentials)
        (*p) *= *p_obs ;
      delete p_obs;
      return message;
    }

    // ------------- FORWARD ------------- //
  public:
    Result filtering(const Matrix& obs, Evaluator *evaluator = nullptr) {
      // Run forward
      forward(obs);

      // Calculate mean and cpp
      Result result;
      for(auto &message : alpha){
        result.mean.appendColumn(message.mean());
        result.cpp.append(message.cpp());
      }

      // Evaluate
      result.ll.append(alpha.back().log_likelihood());
      if(evaluator){
        result.append(evaluator->evaluate(result.cpp));
      }

      return result;
    }

    void forward(const Matrix& obs){
      alpha.clear();
      alpha_predict.clear();
      for (size_t i=0; i<obs.ncols(); i++)
        oneStepForward(obs.getColumn(i));
    }

    void oneStepForward(const Vector& obs) {
      // Predict step
      if (alpha_predict.empty()) {
        Message message(max_components);
        message.add_potential(model->getNoChangePotential());
        message.add_potential(model->getChangePotential());
        alpha_predict.push_back(message);
      }
      else {
        alpha_predict.push_back(predict(alpha.back()));
      }
      // Update step
      alpha.push_back(update(alpha_predict.back(), obs));
    }

    // ------------- BACKWARD ------------- //
    void backward(const Matrix& obs, size_t idx = 0, size_t steps = 0){
      // Start from column "idx" and go back for "steps" steps
      if(steps == 0 ){
        steps = obs.ncols();
        idx = obs.ncols()-1;
      }
      beta.clear();
      Message message(max_components);
      for(size_t t = 0; t < steps; ++t, --idx){
        double delta_log_c = 0;
        if(!beta.empty()){
          // Predict for case s_t = 1, calculate constant only
          Message temp = beta.back();
          for(Potential *p : temp.potentials){
            (*p) *= *model->prior;
          }
          delta_log_c = model->log_p1 + temp.log_likelihood();

          // Update :
          message = update(beta.back(), obs.getColumn(idx));
          for(Potential *p : message.potentials){
            p->log_c += model->log_p0;
          }
        }
        Potential *p = model->obs2Potential(obs.getColumn(idx));
        p->log_c += delta_log_c;
        message.add_potential(p);
        message.prune(max_components);
        beta.push_back(message);
      }
      std::reverse(beta.begin(), beta.end());
    }


    Result smoothing(const Matrix& obs, Evaluator *evaluator = nullptr) {
      // Run Forward - Backward
      forward(obs);
      backward(obs);

      // Calculate Smoothed density
      Result result;
      for(size_t i=0; i < obs.ncols(); ++i) {
        Message gamma = alpha_predict[i] * beta[i];
        result.mean.appendColumn(gamma.mean());
        result.cpp.append(gamma.cpp(beta[i].size()));
      }

      // Evaluate
      result.ll.append(alpha.back().log_likelihood());
      if(evaluator){
        result.append(evaluator->evaluate(result.cpp));
      }

      return result;
    }

    Result online_smoothing(const Matrix& obs, size_t lag,
                            Evaluator *evaluator = nullptr){
      if(lag == 0)
        return filtering(obs, evaluator);

      if(lag >= obs.ncols())
        return smoothing(obs, evaluator);

      Result result;
      Message gamma;

      // Go forward
      forward(obs);

      // Run Fixed-Lags for alpha[0:T-lag]
      for(size_t t=0; t <= obs.ncols()-lag; ++t){
        backward(obs, t+lag-1, lag);
        gamma = alpha_predict[t] * beta.front();
        result.mean.appendColumn(gamma.mean());
        result.cpp.append(gamma.cpp(beta.front().size()));
      }

      // Smooth alpha[T-lag+1:T] with last beta.
      for(size_t i = 1; i < lag; ++i){
        gamma = alpha_predict[obs.ncols()-lag+i] * beta[i];
        result.mean.appendColumn(gamma.mean());
        result.cpp.append(gamma.cpp(beta[i].size()));
      }

      // Evaluate
      result.ll.append(alpha.back().log_likelihood());
      if(evaluator){
        result.append(evaluator->evaluate(result.cpp));
      }

      return result;
    }

    Vector compute_ss(const Message &message) {
      Matrix tmp;
      Vector norm_consts;
      for(Potential *p : message.potentials){
        norm_consts.append(p->log_c);
        tmp.appendColumn( p->get_ss() );
      }
      norm_consts = normalizeExp(norm_consts);
      return dot(tmp, norm_consts);
    }

    Result learn_parameters(const Matrix& obs,
                            Evaluator *evaluator = nullptr,
                            size_t max_iter = 100){
      size_t T = obs.ncols();
      size_t min_iter = 20;
      Vector ll;
      Matrix score;

      for(size_t iter = 0; iter < max_iter; ++iter){

        // Forward_backward
        forward(obs);
        backward(obs);

        // Part 1: E-Step
        Vector cpp(T);
        Matrix ss;
        for(size_t i=0; i < T; ++i) {
          Message gamma = alpha_predict[i] * beta[i];
          cpp(i) = gamma.cpp(beta[i].size());
          ss.appendColumn(compute_ss(gamma));
        }
        Vector ss_w = dot(ss, cpp) / sum(cpp);
        double ss_p1 = sum(cpp) / T;

        // Part 2: M-Step
        model->fit(ss_w, ss_p1);

        // Part 3: Evaluate
        ll.append(alpha.back().log_likelihood());
        if(evaluator){
          score.appendRow(evaluator->evaluate(cpp));
        }

        // Part 4: Stop if converged
        std::cout << "iter: " << iter
                  << "\tlog-likelihood : " << ll.last() << std::endl;
        if(iter > 0 ){
          double ll_diff = ll[iter] - ll[iter-1];
          if( ll_diff < 0 ){
            std::cout << "!!! log-likelihood decreased: "
                      << - ll_diff << std::endl;
            break;
          }
          if( iter > min_iter && ( ll_diff < 1e-6)){
            std::cout << "converged.\n";
            break;
          }
        }
      }

      Result result = smoothing(obs);
      result.ll = ll;
      result.score = score;
      return result;
    }


  public:
    Model *model;
    int max_components;

  private:
    std::vector<Message> alpha;
    std::vector<Message> alpha_predict;
    std::vector<Message> beta;

};

#endif //BCPM_H_
