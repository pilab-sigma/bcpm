#include "bcpm_gaussian.hpp"

using namespace std;
using namespace pml;

const double threshold = 0.99;
const size_t window = 1;

void test_gaussian_em(const std::string &base_dir){

  double p1 = 0.01;
  double mu = 3;
  double sigma = 2;
  size_t length = 200;

  // Generate data:
  GaussianModel model(mu, sigma, p1);

  // Generate Sequence
  auto data = model.generateData(length);
  data.saveTxt("/tmp");

  // Estimate with true parameters
  ForwardBackward fb(&model);
  auto result = fb.smoothing(data.obs);
  result.saveTxt("/tmp");


  // Random init a model
  GaussianModel init_model(Uniform(0, 5).rand(), Uniform(0, 5).rand(), p1);
  GaussianModel em_model = init_model;

  ForwardBackward fb_em(&em_model);
  result = fb_em.smoothing(data.obs);
  result.saveTxt("/tmp/initial");

  result = fb_em.learn_parameters(data.obs);
  result.saveTxt("/tmp/final");

  std::cout << "-----------\n";
  std::cout << "True model:\n";
  model.print();
  std::cout << "-----------\n";
  std::cout << "EM(initial) model:\n";
  init_model.print();
  std::cout << "-----------\n";
  std::cout << "EM(final) model:\n";
  em_model.print();
  std::cout << "-----------\n";

  cout << "OK.\n";
}

int main(int argc, char *argv[]){

  std::string base_dir = "/tmp";
  if(argc == 2)
    base_dir = argv[1];

  test_gaussian_em(base_dir);

  return 0;
}