#include <bcpm_dm.hpp>
#include <bcpm_pg.hpp>
#include "bcpm_comp.hpp"

using namespace std;
using namespace pml;


const double threshold = 0.99;
const size_t window = 1;

size_t length = 100;
double c = 0.01;

double alpha_ = 5;
double a_ = 10;
double b_ = 5;

size_t M = 5;
size_t N = 3;

void test_comp() {
  cout << "test_compound...\n";
  Vector alpha = Vector::ones(M)*alpha_;
  Vector a = Vector::ones(N)*a_;
  Vector b = Vector::ones(N)*b_;

  // Generate model:
  COMP_Model model(alpha, a, b, c);

  // Generate Sequence
  auto data = model.generateData(length);
  data.saveTxt("/tmp");

  Evaluator evaluator(data.cps, threshold, window);

  ForwardBackward fb(&model);
  std::cout << "Filtering...\n";
  auto result =  fb.filtering(data.obs, &evaluator);
  cout << result.cpp << endl << sum(result.cpp) << endl;

  result.saveTxt("/tmp/filtering");
}