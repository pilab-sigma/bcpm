//
// Created by cagatay on 11.03.2017.
//

#include <bcpm_comp.hpp>
#include "../old/bcpm_compound.hpp"


using namespace std;
using namespace pml;


const double threshold = 0.99;
const size_t window = 1;
const size_t lag = 10;

size_t length = 200;
double c = 0.01;

double alpha_ = 5;
double a_ = 10;
double b_ = 5;

size_t M = 5;
size_t N = 3;

void check_results(Result &r1, old::Result &r2){

  assert(r1.cpp.equals(r2.cpp));
  assert(r1.mean.equals(r2.mean));
  assert(r1.ll.equals(r2.ll));
  assert(r1.score.equals(r2.score));
}

void test_coupled() {
  cout << "Test Compound Model...\n";
  Vector alpha = Vector::ones(M) * alpha_;
  Vector a = Vector::ones(N) * a_;
  Vector b = Vector::ones(N) * b_;

  // Generate model:
  COMP_Model model(alpha, a, b, c);
  ForwardBackward fb(&model);

  // Generate Old Model:
  old::Model model_old(c, old::Potential(alpha, a, b));

  // Generate Sequence
  auto data = model.generateData(length);

  // Test Filtering
  std::cout << "\tfiltering...\n";
  auto result = fb.filtering(data.obs);
  auto result_old = model_old.filtering(data.obs);
  check_results(result, result_old);

  // Test Smoothing
  std::cout << "\tsmoothing...\n";
  result = fb.smoothing(data.obs);
  result.saveTxt("/tmp/result");
  result_old = model_old.smoothing(data.obs);
  result_old.saveTxt("/tmp/result_old");
  check_results(result, result_old);

  // Test Online Smoothing
  std::cout << "\tonline smoothing...\n";
  result = fb.online_smoothing(data.obs, lag);
  result.saveTxt("/tmp/result");
  result_old = model_old.online_smoothing(data.obs, lag);
  result_old.saveTxt("/tmp/result_old");
  check_results(result, result_old);


  std::cout << "done.\n\n";
}



int main() {
  test_coupled();
  return 0;
}