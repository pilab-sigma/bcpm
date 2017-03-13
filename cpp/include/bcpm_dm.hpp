#ifndef BCPM_DM_H_
#define BCPM_DM_H_

#include "bcpm.hpp"


class DirichletPotential : public Potential {

  public:
    DirichletPotential(size_t K = 0, double log_c_ = 0) : Potential(log_c_) {
      alpha = Vector::ones(K);
    }

    DirichletPotential(const Vector& alpha_, double log_c_ = 0)
        : Potential(log_c_), alpha(alpha_) {}

    DirichletPotential(const DirichletPotential &that)
        : Potential(that.log_c), alpha(that.alpha) {}

    static DirichletPotential rand_gen(size_t K, double precision = 1){
      Vector alpha = normalize(Uniform().rand(K)) * precision;
      return DirichletPotential(alpha);
    }

  public:

    Potential* clone(double delta_log_c = 0) const override {
      return new DirichletPotential(alpha, log_c + delta_log_c);
    }

    void operator*=(const Potential &p) override {
      DirichletPotential *pp = (DirichletPotential*) &p;
      double delta_log_c = gammaln(sum(alpha)) - sum(gammaln(alpha)) +
                           gammaln(sum(pp->alpha)) - sum(gammaln(pp->alpha)) +
                           sum(gammaln(alpha + pp->alpha-1)) -
                           gammaln(sum(alpha + pp->alpha -1));
      alpha += pp->alpha - 1;
      log_c += pp->log_c + delta_log_c;
    }

    Potential* operator*(const Potential &p) const override {
      DirichletPotential *result = new DirichletPotential(*this);
      result->operator*=(p);
      return result;
    }

  public:
    Vector rand() const override {
      return Dirichlet(alpha).rand();
    }

    Vector mean() const override {
      return normalize(alpha);
    }

    Vector get_ss() const override{
      return psi(alpha) - psi(sum(alpha));
    }


    void print() const{
      std::cout << alpha << " log_c:" << log_c << std::endl;
    }

    void fit(const Vector &ss, double precision = 0) {
      alpha = Dirichlet::fit(ss, precision).alpha;
    }

  public:
    Vector alpha;
};



class DM_Model: public Model {

  public:
    DM_Model(const Vector &alpha, double p1_,
             bool fixed_precision = false) : Model( p1_) {
      prior = new DirichletPotential(alpha);
      precision = fixed_precision ? sum(alpha) : 0;
    }

    Vector randState() const override {
      return prior->rand();
    }

    Vector randObservation(const Vector &state) const override {
      return Multinomial(state, 20).rand();
    }

    Potential* getPrior() override {
      return prior;
    }

    void fit(const Vector &ss, double p1_new) override {
      ((DirichletPotential*)prior)->fit(ss, precision);
      set_p1(p1_new);
    }

    Potential* obs2Potential(const Vector& obs) const override{
      double log_c = gammaln(sum(obs)+1) - gammaln(sum(obs)+obs.size());
      return new DirichletPotential(obs+1, log_c);
    }

    void saveTxt(const std::string &filename) const override{
      const int txt_precision = 10;
      Vector temp;
      temp.append(p1);
      temp.append(((DirichletPotential *)prior)->alpha);
      temp.append(precision);
      temp.saveTxt(filename, txt_precision);
    }

    void loadTxt(const std::string &filename) override {
      Vector temp = Vector::loadTxt(filename);
      set_p1(temp(0));
      prior = new DirichletPotential(temp.getSlice(1, temp.size()-1));
      precision = temp.last();
    }

    void print() const override{
      std::cout << "DM_Model: \n\talpha = "
                << ((DirichletPotential*)prior)->alpha << std::endl
                << "\tp1 = " << p1 << std::endl
                << "\tfixed_precision = " << (int)(precision == 0) << "\n";
    }

  public:
    DirichletPotential* prior;
    double precision;
};

#endif //BCPM_DM_H_