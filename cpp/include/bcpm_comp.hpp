//
// Created by cagatay on 11.03.2017.
//

#ifndef BCPM_BCPM_COMP_H
#define BCPM_BCPM_COMP_H

#include "bcpm.hpp"

class CompoundPotential : public Potential {

  public:
    CompoundPotential(const Vector &alpha_, const Vector &a_,
                      const Vector &b_, double log_c_ = 0)
        : Potential(log_c_), alpha(alpha_), a(a_), b(b_){
      M = alpha.size();
      N = a.size();
    }

    CompoundPotential(size_t M_=0, size_t N_=0, double log_c_=0)
        : Potential(log_c_), M(M_), N(N_) {
      if(M_ > 0)
        alpha = Vector::ones(M_);
      if (N_ > 0) {
        a = Vector::ones(N_)*10;
        b = Vector::ones(N_);
      }
    }

    CompoundPotential(const CompoundPotential& that)
        : Potential(that.log_c), alpha(that.alpha),
          a(that.a), b(that.b), M(that.M), N(that.N){}

  public:

    Potential* clone() const override {
      return new CompoundPotential(*this);
    }

    void operator*=(const Potential &p) override {
      CompoundPotential* pp = (CompoundPotential*) &p;
      log_c += pp->log_c;
      if( M > 0 ){
        log_c += gammaln(sum(alpha)) - sum(gammaln(alpha)) +
                 gammaln(sum(pp->alpha)) - sum(gammaln(pp->alpha)) +
                 sum(gammaln(alpha + pp->alpha-1)) -
                 gammaln(sum(alpha + pp->alpha -1));
        alpha += pp->alpha - 1;
      }
      if( N > 0 ){
        Vector a_ = a + pp->a - 1;
        Vector b_ = (b * pp->b) / (b + pp->b);
        log_c += sum(gammaln(a_)) + sum(a_ * log(b_))
                 - sum(gammaln(a)) - sum(a * log(b))
                 - sum(gammaln(pp->a)) - sum(pp->a * log(pp->b));
        this->a  = a_;
        this->b  = b_;
      }
    }

    Potential* operator*(const Potential &p) const override {
      CompoundPotential *result = new CompoundPotential(*this);
      result->operator*=(p);
      return result;
    }

  public:
    Vector rand() const override {
      Vector state;
      if(M > 0)
        state = Dirichlet(alpha).rand();
      for(size_t i=0; i < N; ++i)
        state.append(Gamma(a[i], b[i]).rand(1));
      return state;
    }

    Vector mean() const override {
      Vector mean;
      if(M > 0)
        mean = normalize(alpha);
      if(N > 0)
        mean.append(a*b);
      return mean;
    }

    Vector get_ss() const override {
      Vector ss;
      if(M > 0)
        ss = psi(alpha) - psi(sum(alpha));
      if(N > 0) {
        ss.append(a * b);
        ss.append(psi(a) + log(b));
      }
      return ss;
    }

    void print() const  {
      std::cout << "alpha:" << alpha << "  a:" << a
      << "  b:" << b << "  log_c: " << log_c << std::endl;
    }


    void fit(const Vector &ss, double scale = 0){
      if(M > 0)
        alpha = Dirichlet::fit(ss.getSlice(0, M), M).alpha;
      for(size_t i=0; i < N; ++i){
        Gamma g = Gamma::fit(ss[M+i], ss[M+N+i], 1);
        a[i] = g.a;
        b[i] = g.b;
      }
    }

  public:
    Vector alpha;
    Vector a;
    Vector b;
    size_t M; // = alpha.size()
    size_t N; // = a.size()
};


class COMP_Model : public Model{

  public:
    COMP_Model():Model(1){}
    COMP_Model(Vector alpha, Vector a, Vector b, double p1_)
        :Model(p1_) {
      prior = new CompoundPotential(alpha, a, b);
      M = alpha.size();
      N = a.size();
    }

    COMP_Model(const COMP_Model &model) : Model(model.p1){
      prior = (CompoundPotential*) model.prior->clone();
      M = model.M;
      N = model.N;
    }

    ~COMP_Model(){
      delete prior;
    }

    COMP_Model& operator=(const COMP_Model &model){
      if(prior)
        delete prior;
      set_p1(model.p1);
      prior = (CompoundPotential*) model.prior->clone();
      M = model.M;
      N = model.N;
      return *this;
    }

    const Potential* getPrior() override {
      return prior;
    }

    Vector randState() const override {
      return prior->rand();
    }

    Vector randObservation(const Vector &state) const override {
      Vector result;
      if (M > 0)
        result = Multinomial(state.getSlice(0, M), 100).rand();
      for(size_t i = 0; i < N; ++i)
        result.append(Poisson(state[M+ i]).rand());
      return result;
    }

    Potential* obs2Potential(const Vector& obs) const override{
      CompoundPotential *p = new CompoundPotential(M, N);
      if(M > 0){
        Vector obs_alpha = obs.getSlice(0, M);
        p->alpha =  obs_alpha + 1;
        p->log_c = gammaln(sum(obs_alpha)+1) - gammaln(sum(obs_alpha)+M);
      }
      if(N > 0) {
        p->a = obs.getSlice(M, M+N) +1;
        p->b = Vector::ones(N);
      }
      return p;
    }

    void fit(const Vector &ss, double p1_new) override {
      prior->fit(ss);
      set_p1(p1_new);
    }

    void saveTxt(const std::string &filename) const override {
      const int precision = 10;
      Vector temp;
      temp.append(p1);
      temp.append(M);
      temp.append(N);
      temp.append(prior->alpha);
      temp.append(prior->a);
      temp.append(prior->b);
      temp.saveTxt(filename, precision);
    }

    void loadTxt(const std::string &filename) override {
      std::cout << filename << std::endl;
      Vector temp = Vector::loadTxt(filename);
      std::cout << temp << std::endl;
      set_p1(temp(0));
      M = temp(1);
      N = temp(2);
      Vector alpha = temp.getSlice(3, M+3);
      Vector a = temp.getSlice(M+3, M+N+3);
      Vector b = temp.getSlice(M+N+3, M+N+N+3);
      prior = new CompoundPotential(alpha, a, b);
    }

    void print() const override{
      std::cout << "COMP_Model:\n";
      std::cout << "alpha = " << prior->alpha << "\ta = " << prior->a
      << "\tb = " << prior->b << "\tp1 = " << p1 << std::endl;
    }

  public:
    size_t M;
    size_t N;
    CompoundPotential *prior;
};


#endif //BCPM_BCPM_COMP_H
