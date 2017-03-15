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

void test_comp_dm() {
  cout << "compound & dm comparison ...\n";
  Vector alpha = Vector::ones(M)*alpha_;
  Vector a = Vector::ones(0);
  Vector b = Vector::ones(0);

  COMP_Model comp_model(alpha, a, b, c);
  auto data = comp_model.generateData(length);
  data.saveTxt("/tmp");

  ForwardBackward comp_fb(&comp_model);
  std::cout << "Compound filtering...\n";
  auto result_comp =  comp_fb.filtering(data.obs);
  result_comp.saveTxt("/tmp/filtering_comp");


  DM_Model dm_model(alpha, c);
  ForwardBackward dm_fb(&dm_model);
  std::cout << "DM filtering...\n";
  auto result_dm =  dm_fb.filtering(data.obs);
  result_dm.saveTxt("/tmp/filtering_dm");

  std::cout << "likelihoods:\t" << result_comp.ll << "\t" << result_dm.ll << std::endl;
  std::cout << "difference of cpps:\n" << result_comp.cpp-result_dm.cpp << std::endl;

}

void test_comp_pg() {
  cout << "\ncompound & pg comparison ...\n";
  Vector alpha = Vector::ones(0);
  Vector a = Vector::ones(1)*a_;
  Vector b = Vector::ones(1)*b_;

  COMP_Model comp_model(alpha, a, b, c);
  auto data = comp_model.generateData(length);
  data.saveTxt("/tmp");

  ForwardBackward comp_fb(&comp_model);
  std::cout << "Compound filtering...\n";
  auto result_comp =  comp_fb.filtering(data.obs);
  result_comp.saveTxt("/tmp/filtering_comp");


  PG_Model pg_model(a_, b_, c);
  ForwardBackward pg_fb(&pg_model);
  std::cout << "PG filtering...\n";
  auto result_pg =  pg_fb.filtering(data.obs);
  result_pg.saveTxt("/tmp/filtering_pg");

  std::cout << "likelihoods:\t" << result_comp.ll << "\t" << result_pg.ll << std::endl;
  std::cout << "difference of cpps:\n" << result_comp.cpp-result_pg.cpp << std::endl;

}

int main(){

  test_comp();

  // test_comp_dm();

  // test_comp_pg();

  return 0;
}
