#include <cassert>

#include "bcpm_dm.hpp"

using namespace std;
using namespace pml;

/*
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
  ForwardBackward fb(&model);

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
 */

void test_potential() {
  Vector alpha = {1,2,3,4,5};
  double log_c = -0.1;

  DirichletPotential *dm = new DirichletPotential(alpha, log_c);
  assert(alpha.equals(dm->alpha));
  assert(fequal(log_c, dm->log_c));

  DirichletPotential *dm2  = new DirichletPotential(*dm);
  assert(dm->alpha.equals(dm2->alpha));
  assert(fequal(dm->log_c, dm2->log_c));
  assert( dm != dm2 );

  DirichletPotential *dm3  = (DirichletPotential *) dm2->clone(-0.1);
  assert(dm2->alpha.equals(dm3->alpha));
  assert(fequal(dm2->log_c - 0.1 , dm3->log_c));
  assert( dm2 != dm3 );

  delete dm;
  delete dm2;
  delete dm3;
}

void test_message() {

}

/*
void test_fb(){

  const double p1 = 0.01;
  const size_t K = 5;
  const double precision = K;
  const bool fixed_precision = false;
  const size_t length = 120;
  const double threshold = 0.99;
  const size_t window = 1;

  // Generate Model
  Vector alpha = normalize(Vector::ones(K)) * precision;
  DM_Model model(alpha, p1, fixed_precision);

  // Generate Sequence
  auto data = model.generateData(length);
  data.saveTxt("/tmp");

  // Evaluator
  Evaluator evaluator(data.cps, threshold, window);

  // Forward - Backward
  ForwardBackward fb(&model);

  // Filtering
  std::cout << "Filtering...\n";
  auto result =  fb.filtering(data.obs, &evaluator);
  result.saveTxt("/tmp/filtering");
}
*/

int main(){
  test_potential();
  return 0;
}