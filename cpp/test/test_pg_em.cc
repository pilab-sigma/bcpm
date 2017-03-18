#include "bcpm_pg.hpp"

using namespace std;
using namespace pml;

const double threshold = 0.99;
const size_t window = 1;

void test_pg_em(const std::string &base_dir){
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
  data.saveTxt(path_join({base_dir, "data"}));

  Evaluator evaluator(data.cps, threshold, window);

  // Smoothing with true parameters
  ForwardBackward fb(&model);
  Result result = fb.smoothing(data.obs, &evaluator);
  data.saveTxt(path_join({base_dir, "true"}));

  // Generate random model for EM
  double init_p1 = 0.001;
  double init_a = Uniform(0, 10).rand();
  double init_b = 1;

  PG_Model init_model(init_a, init_b, init_p1, fixed_scale);
  PG_Model em_model = init_model;
  ForwardBackward fb_em(&em_model);

  // Run initial model:
  result = fb_em.smoothing(data.obs, &evaluator);
  data.saveTxt(path_join({base_dir, "initial"}));

  // Run EM:
  result = fb_em.learn_parameters(data.obs, &evaluator);
  data.saveTxt(path_join({base_dir, "final"}));

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

  cout << "done.\n";
}

int main(int argc, char *argv[]){

  std::string base_dir = "/tmp";
  if(argc == 2)
    base_dir = argv[1];

  test_pg_em(base_dir);

  return 0;
}