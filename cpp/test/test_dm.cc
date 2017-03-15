#include <cassert>

#include "bcpm_dm.hpp"

using namespace std;
using namespace pml;


void test_dm(const string &base_dir){
  cout << "test_dm...\n";

  const double p1 = 0.01;
  const size_t K = 5;
  const double precision = K;
  const bool fixed_precision = false;
  const size_t length = 200;
  const double threshold = 0.99;
  const size_t window = 1;
  const size_t lag = 10;

  // Generate Model
  const Vector alpha = normalize(Vector::ones(K)) * precision;
  DM_Model model(alpha, p1, fixed_precision);

  // Generate Sequence
  auto data = model.generateData(length);
  data.saveTxt(path_join({base_dir, "data"}));

  Evaluator evaluator(data.cps, threshold, window);

  // Generate Forward-Backward
  ForwardBackward fb(&model);

  // Filtering
  std::cout << "Filtering...\n";
  auto result =  fb.filtering(data.obs, &evaluator);
  result.saveTxt(path_join({base_dir, "filtering"}));

  // Smoothing
  std::cout << "Smoothing...\n";
  result = fb.smoothing(data.obs, &evaluator);
  result.saveTxt(path_join({base_dir, "smoothing"}));

  // Fixed Lag
  std::cout << "Online smoothing...\n";
  result = fb.online_smoothing(data.obs, lag, &evaluator);
  result.saveTxt(path_join({base_dir, "online_smoothing"}));

  cout << "OK.\n";

  return;
}

void visualize(const string &python_exec){
  const std::string cmd = python_exec + " ../visualize/test_dm.py";
  if(system(cmd.c_str()))
    std::cout <<"visualization error...\n";
}

int main(int argc, char *argv[]){

  test_dm("/tmp");

  // Visualize
  std::string python_exec = "python";
  if(argc == 2)
    python_exec = argv[1];
  visualize(python_exec);

  return 0;
}