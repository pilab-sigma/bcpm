#include <cassert>

#include <bcpm_dm.hpp>
#include <bcpm_pg.hpp>
#include <bcpm_comp.hpp>

using namespace std;
using namespace pml;

void check_results(Result &r1, Result &r2){
  assert(r1.mean.equals(r2.mean));
  assert(r1.cpp.equals(r2.cpp));
  assert(r1.ll.equals(r2.ll));
  assert(r1.score.equals(r2.score));
}

void test_dm() {

  std::cout << "Testing DM Model...\n";

  const size_t M = 5;
  const size_t length = 200;
  const double p1 = 0.01;
  const size_t lag = 10;

  // Generate DM Model
  const Vector alpha = Vector::ones(M);
  DM_Model model_dm(alpha, p1);
  ForwardBackward fb_dm(&model_dm);

  // Generate Coupled Model with DM component only
  COMP_Model model_comp(alpha, Vector(), Vector(), p1);
  ForwardBackward fb_comp(&model_comp);

  // Generate Data
  auto data = model_dm.generateData(length);

  // Test Filtering
  std::cout << "\tfiltering...\n";
  auto result_dm =  fb_dm.filtering(data.obs);
  auto result_comp =  fb_comp.filtering(data.obs);
  check_results(result_dm, result_comp);

  // Test Smoothing
  std::cout << "\tsmoothing...\n";
  result_dm =  fb_dm.smoothing(data.obs);
  result_comp =  fb_comp.smoothing(data.obs);
  check_results(result_dm, result_comp);

  // Test Online Smoothing
  std::cout << "\tonline smoothing...\n";
  result_dm =  fb_dm.online_smoothing(data.obs, lag);
  result_comp =  fb_comp.online_smoothing(data.obs, lag);
  check_results(result_dm, result_comp);

  std::cout << "done.\n\n";
}

void test_pg() {

  std::cout << "Testing PG Model...\n";

  const size_t length = 200;
  const double p1 = 0.01;
  const size_t lag = 10;

  // Generate DM Model
  Vector a = Vector(1, 10);
  Vector b = Vector(1, 5);

  PG_Model model_pg(a[0], b[0] ,p1);
  ForwardBackward fb_pg(&model_pg);

  // Generate Coupled Model with DM component only
  COMP_Model model_comp(Vector(), a, b, p1);
  ForwardBackward fb_comp(&model_comp);

  // Generate Data
  auto data = model_pg.generateData(length);

  // Test Filtering
  std::cout << "\tfiltering...\n";
  auto result_pg =  fb_pg.filtering(data.obs);
  auto result_comp =  fb_comp.filtering(data.obs);
  check_results(result_pg, result_comp);

  // Test Smoothing
  std::cout << "\tsmoothing...\n";
  result_pg =  fb_pg.smoothing(data.obs);
  result_comp =  fb_comp.smoothing(data.obs);
  check_results(result_pg, result_comp);

  // Test Online Smoothing
  std::cout << "\tonline smoothing...\n";
  result_pg =  fb_pg.online_smoothing(data.obs, lag);
  result_comp =  fb_comp.online_smoothing(data.obs, lag);
  check_results(result_pg, result_comp);

  std::cout << "done.\n\n";
}

int main(int argc, char *argv[]){

  test_dm();
  test_pg();

  return 0;
}
