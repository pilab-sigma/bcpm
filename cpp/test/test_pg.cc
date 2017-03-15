#include "bcpm_pg.hpp"

using namespace std;
using namespace pml;


void test_pg(const string &base_dir){
  cout << "test_pg...\n";

  const double threshold = 0.99;
  const size_t window = 1;
  const double c = 0.01;
  const double a = 10;
  const double b = 1;
  const size_t length = 1000;

  // Generate model:
  PG_Model model(a, b, c);

  // Generate sequence
  auto data = model.generateData(length);
  data.saveTxt("/tmp/data");

  // Estimate with true parameters
  ForwardBackward fb(&model);

  Evaluator evaluator(data.cps, threshold, window);

  std::cout << "Filtering...\n";
  auto result = fb.filtering(data.obs, &evaluator);
  result.saveTxt(path_join({base_dir, "filtering"}));

  std::cout << "Smoothing...\n";
  result = fb.smoothing(data.obs, &evaluator);
  result.saveTxt(path_join({base_dir, "smoothing"}));

  std::cout << "Online smoothing...\n";
  size_t lag = 10;
  result = fb.online_smoothing(data.obs, lag, &evaluator);
  result.saveTxt(path_join({base_dir, "online_smoothing"}));

  cout << "OK.\n";
}

void visualize(const string &python_exec){
  const std::string cmd = python_exec + " ../visualize/test_pg.py";
  if(system(cmd.c_str()))
    std::cout <<"visualization error...\n";
}


int main(int argc, char *argv[]){

  test_pg("/tmp");

  // Visualize
  std::string python_exec = "python";
  if(argc == 2)
    python_exec = argv[1];
  visualize(python_exec);

  return 0;
}