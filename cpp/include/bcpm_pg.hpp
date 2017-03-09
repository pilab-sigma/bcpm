#ifndef BCPM_PG_H_
#define BCPM_PG_H_

#include "bcpm.hpp"


class GammaPotential : public Potential {

  public:
    GammaPotential(double a_ = 1, double b_ = 1, double log_c_ = 0)
        : Potential(log_c_), a(a_), b(b_){}

  public:
    void operator*=(const GammaPotential &other){
      *this = *this * other;
    }

    friend GammaPotential operator*(const GammaPotential &g1,
                                    const GammaPotential &g2) {
      double a = g1.a + g2.a - 1;
      double b = (g1.b * g2.b) / (g1.b + g2.b);
      double log_c = gammaln(a) + a * std::log(b)
                     - gammaln(g1.a) - g1.a * std::log(g1.b)
                     - gammaln(g2.a) - g2.a * std::log(g2.b);
      return GammaPotential(a, b, g1.log_c + g2.log_c + log_c);
    }

    GammaPotential obs2Potential(const Vector& obs) const {
      return GammaPotential(obs.first()+1, 1);
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



class PG_Model : public Model<GammaPotential> {

  public:
    PG_Model(double a, double b, double p1_, bool fixed_scale = false)
        :Model(p1_) {
      prior = GammaPotential(a, b);
      scale = fixed_scale ? b : 0;
    }

    Vector rand(const Vector &state) const override {
      return Poisson(state.first()).rand(1);
    }

    void fit(const Vector &ss, double p1_new) override {
      prior.fit(ss, scale);
      set_p1(p1_new);
    }

    void saveTxt(const std::string &filename) const override {
      const int precision = 10;
      Vector temp;
      temp.append(p1);
      temp.append(prior.a);
      temp.append(prior.b);
      temp.append((int)(scale == 0));
      temp.saveTxt(filename, precision);
    }

    void loadTxt(const std::string &filename) override{
      Vector temp = Vector::loadTxt(filename);
      set_p1(temp(0));
      prior = GammaPotential(temp(1), temp(2));
      scale = temp(3) ? prior.b : 0;
    }

    void print() const override{
      std::cout << "PG_Model:\n";
      std::cout << "a = " << prior.a << "\tb = " << prior.b << "\tp1 = " << p1
      << "\tfixed_scale = " << (int)(scale == 0) << std::endl;
    }

  public:
    double scale;
};

using PG_ForwardBackward = ForwardBackward<GammaPotential>;

#endif //BCPM_PG_H_
