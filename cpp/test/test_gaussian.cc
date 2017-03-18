#include <cassert>

#include "bcpm_gaussian.hpp"

using namespace std;
using namespace pml;


void test_g(const string &base_dir){
  cout << "Testing Gaussian Model...\n";

  double p1 = 0.01;
  double mu = 3;
  double sigma = 2;
  size_t length = 200;
  size_t lag = 10;

  // Generate model:
  GaussianModel model(mu, sigma, p1);

  // Generate Sequence
  auto data = model.generateData(length);
  data.saveTxt(path_join({base_dir, "data"}));

  // Estimate with true parameters
  ForwardBackward fb(&model);

  std::cout << "Filtering...\n";
  auto result = fb.filtering(data.obs);
  result.saveTxt(path_join({base_dir, "filtering"}));

  std::cout << "Smoothing...\n";
  result = fb.smoothing(data.obs);
  result.saveTxt(path_join({base_dir, "smoothing"}));

  std::cout << "Online smoothing...\n";
  result = fb.online_smoothing(data.obs, lag);
  result.saveTxt(path_join({base_dir, "online_smoothing"}));

  cout << "done.\n\n"
       << "For visualization run command:\n\n"
       << "python ../visualize/test_pg.py " + base_dir << std::endl;

  return;
}


int main(int argc, char *argv[]){

  std::string base_dir = "/tmp";
  if(argc == 2)
    base_dir = argv[1];

  test_g(base_dir);

  return 0;
}