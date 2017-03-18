#ifndef BCPM_G_H_
#define BCPM_G_H_

#include "bcpm.hpp"

class GaussianPotential : public Potential {
  public:
    GaussianPotential(double mu_ = 0, double sigma_ = 1, double log_c = 0) :
        Potential(log_c), mu(mu_), sigma(sigma_){}

    GaussianPotential(const GaussianPotential &that)
        : Potential(that.log_c), mu(that.mu), sigma(that.sigma) {}

  public:
    Potential* clone() const override {
      return new GaussianPotential(*this);
    }

    void operator*=(const Potential &p) override {
      GaussianPotential *pp = (GaussianPotential*) &p;
      double ss1 = std::pow(sigma ,2);
      double ss2 = std::pow(pp->sigma ,2);
      double K = gsl_ran_gaussian_pdf(mu - pp->mu , std::sqrt(ss1 + ss2));
      mu = (mu * ss2 + pp->mu * ss1 ) / (ss1 + ss2);
      sigma = std::sqrt(( ss1 * ss2 ) / (ss1 + ss2));
      log_c += pp->log_c + K;
    }

    Potential* operator*(const Potential &p) const override {
      GaussianPotential *result = new GaussianPotential(*this);
      result->operator*=(p);
      return result;
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


class GaussianModel: public Model {

  public:
    GaussianModel(double mu, double sigma, double p1_) : Model(p1_){
      prior = new GaussianPotential(mu, sigma);
    }

    GaussianModel(const GaussianModel& that) : Model(that.p1){
      prior = (GaussianPotential*) that.prior->clone();
    }

    ~GaussianModel(){
      delete prior;
    }

    GaussianModel& operator=(const GaussianModel &that) {
      if(prior)
        delete prior;
      set_p1(that.p1);
      prior = (GaussianPotential*) that.prior->clone();
      return *this;
    }

  public:

    const Potential* getPrior() override {
      return prior;
    }

    Vector randState() const override {
      return prior->rand();
    }

    Vector randObservation(const Vector &state) const override {
      return Gaussian(state.first()).rand(1);
    }

    Potential* obs2Potential(const Vector& obs) const override{
      return new GaussianPotential(obs.first());
    }

    void fit(const Vector &ss, double p1_new) override {
      prior->fit(ss);
      set_p1(p1_new);
    }

    void saveTxt(const std::string &filename) const override {
      const int precision = 10;
      Vector temp;
      temp.append(prior->mu);
      temp.append(prior->sigma);
      temp.saveTxt(filename, precision);
    }

    void loadTxt(const std::string &filename) override{
      Vector temp = Vector::loadTxt(filename);
      prior = new GaussianPotential(temp(0), temp(1));
    }

    void print() const override{
      std::cout << "G_Model: \n";
      std::cout << "\tmu = " << prior->mu << "\tsigma = " << prior->sigma
      << "\tp1 = " << p1 << std::endl;
    }

  public:
    GaussianPotential *prior;

};

#endif  //BCPM_G_H_