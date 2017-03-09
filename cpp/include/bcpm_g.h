#ifndef BCPM_G_H_
#define BCPM_G_H_

#include "bcpm.hpp"

class GaussianPotential : public Potential {
  public:
    GaussianPotential(double mu_ = 0, double sigma_ = 1, double log_c = 0) :
        Potential(log_c), mu(mu_), sigma(sigma_){}

  public:
    void operator*=(const GaussianPotential &other){
      *this = *this * other;
    }

    friend GaussianPotential operator*(const GaussianPotential &g1,
                                       const GaussianPotential &g2) {
      double ss1 = std::pow(g1.sigma ,2);
      double ss2 = std::pow(g2.sigma ,2);
      double mu = (g1.mu * ss2 + g2.mu * ss1 ) / (ss1 + ss2);
      double sigma = std::sqrt(( ss1 * ss2 ) / (ss1 + ss2));
      double K = gsl_ran_gaussian_pdf(g1.mu - g2.mu , std::sqrt(ss1 + ss2));
      return GaussianPotential(mu, sigma, K + g1.log_c + g2.log_c);
    }

    GaussianPotential obs2Potential(const Vector& obs) const {
      return GaussianPotential(obs.first());
    }

  public:
    Vector rand() const override {
      return Gaussian(mu, sigma).rand(1);
    }

    Vector mean() const override {
      return Vector(1, mu);
    }

    Vector get_ss() const override{
      return Vector({mu, std::pow(sigma,2)});
    }

    void fit(const Vector &ss) {
      mu = ss(0);
      sigma = std::sqrt(ss(1));
    }

  public:
    double mu, sigma;
};


class G_Model: public Model<GaussianPotential> {

  public:
    G_Model(double mu, double sigma, double p1_)
        : Model(p1_){
      prior = GaussianPotential(mu, sigma);
    }

    Vector rand(const Vector &state) const override {
      return Gaussian(state.first()).rand(1);
    }

    void fit(const Vector &ss, double p1_new) override {
      prior.fit(ss);
      set_p1(p1_new);
    }

    void saveTxt(const std::string &filename) const override {
      const int precision = 10;
      Vector temp;
      temp.append(prior.mu);
      temp.append(prior.sigma);
      temp.saveTxt(filename, precision);
    }

    void loadTxt(const std::string &filename) override{
      Vector temp = Vector::loadTxt(filename);
      prior = GaussianPotential(temp(0), temp(1));
    }

    void print() const override{
      std::cout << "G_Model: \n";
      std::cout << "\tmu = " << prior.mu << "\tsigma = " << prior.sigma
      << "\tp1 = " << p1 << std::endl;
    }
};

using G_ForwardBackward = ForwardBackward<GaussianPotential>;

#endif  //BCPM_G_H_