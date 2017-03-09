#ifndef BCPM_H_
#define BCPM_H_

#include <pml/pml.hpp>

using namespace pml;

// ----------- POTENTIAL ----------- //

class Potential {

  public:
    explicit Potential(double log_c_) : log_c(log_c_) {}

  public:
    bool operator<(const Potential &other) const{
      return this->log_c < other.log_c;
    }

  public:
    virtual Vector rand() const = 0;
    virtual Vector mean() const = 0;
    virtual Vector get_ss() const = 0;

  public:
    double log_c;
};


// ----------- MODEL ----------- //

class Model{

  public:
    struct Data{

      Data(){}

      Data(const std::string &dir){
        obs = Matrix::loadTxt(make_path({dir, "obs.txt"}));
        states = Matrix::loadTxt(make_path({dir, "states.txt"}));
        cps = Vector::loadTxt(make_path({dir, "cps.txt"}));
      }

      void saveTxt(const std::string &dir){
        find_or_create(dir);
        obs.saveTxt(make_path({dir, "obs.txt"}));
        states.saveTxt(make_path({dir, "states.txt"}));
        cps.saveTxt(make_path({dir, "cps.txt"}));
      }

      Matrix obs;
      Matrix states;
      Vector cps;
    };

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
    Data generateData(size_t length){
      Data data;
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
    virtual Potential* obs2Potential(const Vector &obs) = 0;
    virtual Vector rand(const Vector &state) const = 0;
    virtual void fit(const Vector &ss, double p1_new) = 0;
    virtual void saveTxt(const std::string &filename) const = 0;
    virtual void loadTxt(const std::string &filename) = 0;
    virtual void print() const = 0;

  public:
    Potential *prior;
    double p1, log_p1, log_p0;
};

// ----------- Message ----------- //

class Message {

  public:
    size_t size() const {
      return potentials.size();
    }

    void add_potential(Potential &potential){
      potentials.push_back(potential);
    }

    void add_potential(Potential &potential, double log_c){
      potentials.push_back(potential);
      potentials.back()->log_c = log_c;
    }

    friend Message<P> operator*(const Message &m1, const Message &m2){
      Message<P> msg;
      for(Potential *p1 : m1.potentials)
        for(Potential *p2 : m2.potentials)
          msg.add_potential((*p1) * (*p2));
      return msg;
    }

    Vector mean() const {
      Matrix params;
      for(Potential *p: potentials)
        params.appendColumn(p->mean());
      return dot(params, normalizeExp(get_consts()));
    }

    double cpp(int num_cpp = 1) const {
      Vector consts = normalizeExp(get_consts());
      if(num_cpp == 1){
        return consts.last();
      }
      double result = 0;
      for(size_t i = consts.size()-num_cpp; i < consts.size(); ++i)
        result += consts[i];
      return result;
    }


    void prune(size_t max_components){
      /*
      while(size() > max_components){
        // Find mininum no-change element
        auto iter = std::min_element(potentials.begin(), potentials.end()-1);
        // Swap the last two elements to save the order of the change comp.
        std::swap(*(potentials.end()-1), *(potentials.end()-2));
        // Swap last element with the minimum compoment.
        std::swap(*iter, potentials.back());
        // Delete minimum component.
        potentials.pop_back();
      }
       */
    }

    double log_likelihood() const {
      return logSumExp(get_consts());
    }

    Vector get_consts() const {
      Vector consts(potentials.size());
      for(size_t i = 0; i < potentials.size(); ++i){
        consts[i] = potentials[i]->log_c;
      }
      return consts;
    }

  public:
    std::vector<Potential*> potentials;

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
      Message message = prev;
      Vector consts;
      for(Potential *p : message.potentials){
        consts.append(p->log_c);
        p->log_c += model->log_p0;
      }
      message.add_potential(model->getChangePotential(logSumExp(consts)));
      // message.add_potential(model->prior, model->log_p1 + );
      return message;
    }

    /*
    MessageType predict(const Message<P>& prev){
      MessageType message;
      P potential;
      potential.fit(compute_ss(prev));
      double ll = prev.log_likelihood();
      message.add_potential(potential, model->log_p0 + ll);
      message.add_potential(model->prior, model->log_p1 + ll);
      return message;
    }
    */

    MessageType update(const MessageType &prev, const Vector &obs){
      MessageType message = prev;
      for(Potential *p : message.potentials)
        (*p) *= model->obs2Potential(obs);
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
      for (size_t i=0; i<obs.ncols(); i++) {
        oneStepForward(obs.getColumn(i));
        alpha.back().prune(max_components);
      }
    }

    void oneStepForward(const Vector& obs) {
      // Predict step
      if (alpha_predict.empty()) {
        Message<P> message;
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
      MessageType message;
      for(size_t t = 0; t < steps; ++t, --idx){
        double c = 0;
        if(!beta.empty()){
          // Predict for case s_t = 1, calculate constant only
          MessageType temp = beta.back();
          for(Potential *p : temp.potentials){
            (*p) *= model->prior;
          }
          c = model->log_p1 + temp.log_likelihood();

          // Update :
          message = update(beta.back(), obs.getColumn(idx));
          for(Potential *p : message.potentials){
            p->log_c += model->log_p0;
          }
        }
        P pot = model->obs2Potential(obs.getColumn(idx));
        pot.log_c += c;
        message.add_potential(pot);
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
        MessageType gamma = alpha_predict[i] * beta[i];
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
      MessageType gamma;

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

    Vector compute_ss(const MessageType &message) {
      Matrix tmp;
      Vector norm_consts;
      for(auto &potential : message.potentials){
        norm_consts.append(potential.log_c);
        tmp.appendColumn( potential.get_ss() );
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
          MessageType gamma = alpha_predict[i] * beta[i];
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
    ModelType *model;
    int max_components;

  private:
    std::vector<MessageType> alpha;
    std::vector<MessageType> alpha_predict;
    std::vector<MessageType> beta;

};

#endif //BCPM_H_
