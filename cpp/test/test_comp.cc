#include <bcpm_dm.hpp>
#include <bcpm_pg.hpp>
#include "bcpm_comp.hpp"

using namespace std;
using namespace pml;

const size_t M = 5;
const size_t N = 3;

void test_comp(const string &base_dir) {

  const double threshold = 0.99;
  const size_t window = 1;

  const size_t length = 100;
  const double c = 0.01;

  const double alpha_ = 5;
  const double a_ = 10;
  const double b_ = 5;

  const size_t lag = 10;


  cout << "test_compound...\n";
  Vector alpha = Vector::ones(M)*alpha_;
  Vector a = Vector::ones(N)*a_;
  Vector b = Vector::ones(N)*b_;

  // Generate model:
  COMP_Model model(alpha, a, b, c);

  // Generate Sequence
  auto data = model.generateData(length);
  data.saveTxt(path_join({base_dir, "data"}));

  Evaluator evaluator(data.cps, threshold, window);

  ForwardBackward fb(&model);
  std::cout << "Filtering...\n";
  auto result =  fb.filtering(data.obs, &evaluator);
  result.saveTxt(path_join({base_dir, "filtering"}));
  cout << result.cpp << endl << sum(result.cpp) << endl;


  std::cout << "Smoothing...\n";
  result =  fb.smoothing(data.obs, &evaluator);
  result.saveTxt(path_join({base_dir, "smoothing"}));
  cout << result.cpp << endl << sum(result.cpp) << endl;


  std::cout << "Filtering...\n";
  result =  fb.online_smoothing(data.obs, lag, &evaluator);
  result.saveTxt(path_join({base_dir, "online_smoothing"}));
  cout << result.cpp << endl << sum(result.cpp) << endl;

  std::cout << "OK.\n";
}

void visualize(const string &python_exec){
  const std::string cmd = python_exec + " ../visualize/test_comp.py /tmp "
                          + std::to_string(M);
  if(system(cmd.c_str()))
    std::cout <<"visualization error...\n";
}


int main(int argc, char *argv[]){

  test_comp("/tmp");

  // Visualize
  std::string python_exec = "python";
  if(argc == 2)
    python_exec = argv[1];
  visualize(python_exec);

  return 0;
}
