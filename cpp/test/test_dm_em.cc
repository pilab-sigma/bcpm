#include "bcpm_dm.hpp"

using namespace std;
using namespace pml;

const double threshold = 0.99;
const size_t window = 1;


void test_dm_em(const std::string &base_dir){

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
  data.saveTxt(path_join({base_dir, "data"}));

  Evaluator evaluator(data.cps, threshold, window);

  // Estimate with true parameters
  ForwardBackward fb(&model);
  auto result = fb.smoothing(data.obs, &evaluator);
  data.saveTxt(path_join({base_dir, "true"}));

  // Learn parameters
  double c_init = 0.0001;
  DM_Model em_model(Dirichlet(Vector::ones(K)).rand() * precision,
                    c_init, fixed_precision);
  DM_Model em_init_model = em_model;
  ForwardBackward fb_em(&em_model);

  // Run with EM inital
  result = fb_em.smoothing(data.obs, &evaluator);
  data.saveTxt(path_join({base_dir, "initial"}));

  // Learn parameters
  result = fb_em.learn_parameters(data.obs, &evaluator);
  data.saveTxt(path_join({base_dir, "final"}));

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

  cout << "OK.\n";
}

int main(int argc, char *argv[]){

  std::string base_dir = "/tmp";
  if(argc == 2)
    base_dir = argv[1];

  test_dm_em(base_dir);

  return 0;
}