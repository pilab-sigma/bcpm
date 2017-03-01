#include "bcpm.hpp"

using namespace std;
using namespace pml;

const double threshold = 0.99;
const size_t window = 1;

int main(){
  cout << "test_dm...\n";

  size_t K = 5;
  double precision = K;
  bool fixed_precision = false;
  Vector alpha = normalize(Vector::ones(K)) * precision;
  double p1 = 0.01;

  size_t lag = 10;
  size_t length = 500;

  // Generate Model
  DM_Model model(alpha, p1, fixed_precision);

  // Generate Sequence
  auto data = model.generateData(length);
  data.saveTxt("/tmp");

  Evaluator evaluator(data.cps, threshold, window);

  // Generate Forward-Backward
  Matrix mean;
  Vector cpp;
  DM_ForwardBackward fb(&model);

  // Filtering
  std::cout << "Filtering...\n";
  auto result =  fb.filtering(data.obs, &evaluator);
  result.saveTxt("/tmp/filtering");

  // Smoothing
  std::cout << "Smoothing...\n";
  result = fb.smoothing(data.obs, &evaluator);
  result.saveTxt("/tmp/smoothing");

  // Fixed Lag
  std::cout << "Online smoothing...\n";
  result = fb.online_smoothing(data.obs, lag, &evaluator);
  result.saveTxt("/tmp/online_smoothing");

  if(system("anaconda3 ../test/python/test_bcpm_dm.py False")){
    std::cout <<"plotting error...\n";
  }
  cout << "OK.\n";

  return 0;
}