#include "bcpm.hpp"

using namespace std;
using namespace pml;

const double threshold = 0.99;
const size_t window = 1;

int main(){
  cout << "test_pg_em...\n";
  size_t length = 200;
  double p1 = 0.05;
  double a = 10;
  double b = 1;
  bool fixed_scale = true;

  // Generate model:
  PG_Model model(a, b, p1, fixed_scale);

  // Generate Sequence
  auto data = model.generateData(length);
  data.saveTxt("/tmp");

  Evaluator evaluator(data.cps, threshold, window);

  // Smoothing with true parameters
  PG_ForwardBackward fb(&model);
  Result result = fb.smoothing(data.obs, &evaluator);
  result.saveTxt("/tmp");

  // Generate random model for EM
  double init_p1 = 0.001;
  double init_a = Uniform(0, 10).rand();
  double init_b = 1;

  PG_Model init_model(init_a, init_b, init_p1, fixed_scale);
  PG_Model em_model = init_model;
  PG_ForwardBackward fb_em(&em_model);

  // Run initial model:
  result = fb_em.smoothing(data.obs, &evaluator);
  result.saveTxt("/tmp/initial");

  // Run EM:
  result = fb_em.learn_parameters(data.obs, &evaluator);
  result.saveTxt("/tmp/final");

  std::cout << "-----------\n";
  std::cout << "True model:\n";
  model.print();
  std::cout << "-----------\n";
  std::cout << "EM(initial) model:\n";
  init_model.print();
  std::cout << "-----------\n";
  std::cout << "EM(final) model:\n";
  em_model.print();
  std::cout << "-----------\n";

  if(system("anaconda3 ../test/python/test_bcpm_pg.py True")){
    std::cout <<"plotting error...\n";
  }
  cout << "OK.\n";

  return 0;
}
