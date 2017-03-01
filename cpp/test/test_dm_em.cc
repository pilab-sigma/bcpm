#include "bcpm.hpp"

using namespace std;
using namespace pml;

const double threshold = 0.99;
const size_t window = 1;


int main(){

  cout << "test_dm_em...\n";

  size_t K = 5;
  double precision = K;
  bool fixed_precision = true;
  Vector alpha = normalize(Vector::ones(K)) * precision;
  double p1 = 0.1;

  size_t length = 100;

  // Generate model:
  DM_Model model(alpha, p1, fixed_precision);

  // Generate sequence
  auto data = model.generateData(length);
  data.saveTxt("/tmp");

  Evaluator evaluator(data.cps, threshold, window);

  // Estimate with true parameters
  DM_ForwardBackward fb(&model);
  auto result = fb.smoothing(data.obs, &evaluator);
  result.saveTxt("/tmp");

  // Learn parameters
  double c_init = 0.0001;
  DM_Model em_model(DirichletPotential::rand_gen(K, precision).alpha,
                    c_init, fixed_precision);
  DM_Model em_init_model = em_model;
  DM_ForwardBackward fb_em(&em_model);

  // Run with EM inital
  result = fb_em.smoothing(data.obs, &evaluator);
  result.saveTxt("/tmp/initial");

  // Learn parameters
  result = fb_em.learn_parameters(data.obs, &evaluator);
  result.saveTxt("/tmp/final");

  std::cout << "-----------\n";
  std::cout << "True model:\n";
  model.print();
  std::cout << "-----------\n";
  std::cout << "EM(initial) model:\n";
  em_init_model.print();
  std::cout << "-----------\n";
  std::cout << "EM(final) model:\n";
  em_model.print();
  std::cout << "-----------\n";

  if(system("anaconda3 ../test/python/test_bcpm_dm.py True")){
    std::cout <<"plotting error...\n";
  }

  cout << "OK.\n";

  return 0;
}
