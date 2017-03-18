#ifndef BCPM_PG_H_
#define BCPM_PG_H_

#include "bcpm.hpp"


class GammaPotential : public Potential {

  public:
    GammaPotential(double a_ = 1, double b_ = 1, double log_c_ = 0)
        : Potential(log_c_), a(a_), b(b_){}

    GammaPotential(const GammaPotential &that)
        : Potential(that.log_c), a(that.a), b(that.b) {}

  public:

    Potential* clone() const override {
      return new GammaPotential(*this);
    }

    void operator*=(const Potential &p) override {
      GammaPotential *pp = (GammaPotential*) &p;
      double a_ = a + pp->a - 1;
      double b_ = (b * pp->b) / (b + pp->b);
      double log_c_ = gammaln(a_) + a_ * std::log(b_)
                     - gammaln(a) - a * std::log(b)
                     - gammaln(pp->a) - pp->a * std::log(pp->b);
      this->a  = a_;
      this->b  = b_;
      this->log_c += pp->log_c + log_c_;
    }

    Potential* operator*(const Potential &p) const override {
      GammaPotential *result = new GammaPotential(*this);
      result->operator*=(p);
      return result;
    }

  public:
    Vector rand() const override {
      return Gamma(a, b).rand(1);
    }

    Vector mean() const override {
      return Vector(1, a * b);
    }

    Vector get_ss() const override {
      return Vector({a*b, psi(a) + std::log(b)});
    }

    void print() const  {
      std::cout << "a:" << a << "  b:" << b
      << "  log_c: " << log_c << std::endl;
    }

    void fit(const Vector &ss, double scale = 0){
      Gamma g_est = Gamma::fit(ss[0], ss[1], scale);
      a = g_est.a;
      b = g_est.b;
    }

  public:
    double a;  // shape parameter
    double b;  // scale parameter (!!! NOT THE RATE PARAMETER !!!!)
};


class PG_Model : public Model {

  public:
    PG_Model(double a, double b, double p1_, bool fixed_scale = false)
        :Model(p1_) {
      prior = new GammaPotential(a, b);
      scale = fixed_scale ? b : 0;
    }

    PG_Model(const PG_Model &model) : Model(model.p1){
      prior = (GammaPotential*) model.prior->clone();
      scale = model.scale;
    }

    ~PG_Model(){
      delete prior;
    }

    PG_Model& operator=(const PG_Model &model){
      if(prior)
        delete prior;
      set_p1(model.p1);
      prior = (GammaPotential*) model.prior->clone();
      scale = model.scale;
      return *this;
    }


    const Potential* getPrior() override {
      return prior;
    }

    Vector randState() const override {
      return prior->rand();
    }

    Vector randObservation(const Vector &state) const override {
      return Poisson(state.first()).rand(1);
    }

    Potential* obs2Potential(const Vector& obs) const override{
      return new GammaPotential(obs.first()+1, 1);
    }

    void fit(const Vector &ss, double p1_new) override {
      prior->fit(ss, scale);
      set_p1(p1_new);
    }

    void saveTxt(const std::string &filename) const override {
      const int precision = 10;
      Vector temp;
      temp.append(p1);
      temp.append(prior->a);
      temp.append(prior->b);
      temp.append(scale);
      temp.saveTxt(filename, precision);
    }

    void loadTxt(const std::string &filename) override{
      Vector temp = Vector::loadTxt(filename);
      set_p1(temp(0));
      if( prior )
        delete prior;
      prior = new GammaPotential(temp(1), temp(2));
      scale = temp(3);
    }

    void print() const override{
      std::cout << "PG_Model:\n"
                << "a = " << prior->a
                << "\tb = " << prior->b
                << "\tp1 = " << p1
                << "\tfixed_scale = " << (int)(scale == 0) << std::endl;
    }

  public:
    double scale;
    GammaPotential *prior;
};

#endif //BCPM_PG_H_
