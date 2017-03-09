#include "bcpm_pg.hpp"

using namespace std;
using namespace pml;


const double threshold = 0.99;
const size_t window = 1;

int main(){

  cout << "test_pg...\n";
  size_t length = 1000;
  double c = 0.01;
  double a = 10;
  double b = 1;

  // Generate model:
  PG_Model model(a, b, c);

  // Generate sequence
  auto data = model.generateData(length);
  data.saveTxt("/tmp");

  // Estimate with true parameters
  PG_ForwardBackward fb(&model);

  Evaluator evaluator(data.cps, threshold, window);

  std::cout << "Filtering...\n";
  auto result = fb.filtering(data.obs, &evaluator);
  result.saveTxt("/tmp/filtering");

  std::cout << "Smoothing...\n";
  result = fb.smoothing(data.obs, &evaluator);
  result.saveTxt("/tmp/smoothing");


  std::cout << "Online smoothing...\n";
  size_t lag = 10;
  result = fb.online_smoothing(data.obs, lag, &evaluator);
  result.saveTxt("/tmp/online_smoothing");

  if(system("anaconda3 ../test/python/test_bcpm_pg.py False")){
    std::cout <<"plotting error...\n";
  }
  cout << "OK.\n";

  return 0;
}