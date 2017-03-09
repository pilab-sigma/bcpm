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

    static DirichletPotential rand_gen(size_t K, double precision = 1){
      Vector alpha = normalize(Uniform().rand(K)) * precision;
      return DirichletPotential(alpha);
    }

  public:
    void operator*=(const DirichletPotential &p){
      *this = this->operator*(p);
    }


    DirichletPotential operator*(const DirichletPotential &p) const{

      double delta = gammaln(sum(alpha)) - sum(gammaln(alpha)) +
                     gammaln(sum(p.alpha)) - sum(gammaln(p.alpha)) +
                     sum(gammaln(alpha + p.alpha-1)) -
                     gammaln(sum(alpha + p.alpha -1));

      return DirichletPotential(alpha + p.alpha - 1,
                                log_c + p.log_c + delta);
    }

    DirichletPotential obs2Potential(const Vector& obs) const{
      double log_c = gammaln(sum(obs)+1)
                     - gammaln(sum(obs)+obs.size());
      return DirichletPotential(obs+1, log_c);
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



class DM_Model: public Model<DirichletPotential> {

  public:
    DM_Model(const Vector &alpha, double p1_,
             bool fixed_precision = false) : Model( p1_) {
      prior = DirichletPotential(alpha);
      precision = fixed_precision ? sum(alpha) : 0;
    }

    Vector rand(const Vector &state) const override {
      return Multinomial(state, 20).rand();
    }

    void fit(const Vector &ss, double p1_new) override {
      prior.fit(ss, precision);
      set_p1(p1_new);
    }

    void saveTxt(const std::string &filename) const override{
      const int txt_precision = 10;
      Vector temp;
      temp.append(p1);
      temp.append(prior.alpha);
      temp.append(precision == 0);
      temp.saveTxt(filename, txt_precision);
    }

    void loadTxt(const std::string &filename){
      Vector temp = Vector::loadTxt(filename);
      set_p1(temp(0));
      prior = DirichletPotential(temp.getSlice(1, temp.size()-1));
      precision = temp.last() ? sum(prior.alpha) : 0;
    }

    void print() const override{
      std::cout << "DM_Model: \n";
      std::cout << "\talpha = " << prior.alpha << std::endl;
      std::cout << "\tp1 = " << p1 << std::endl;
      std::cout << "\tfixed_precision = " << (int)(precision == 0) << "\n";
    }

  public:
    double precision;
};

using DM_ForwardBackward = ForwardBackward<DirichletPotential>;

#endif //BCPM_DM_H_