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


  cout << "Testing compound model...\n";
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
  std::cout << "\tfiltering...\n";
  auto result =  fb.filtering(data.obs, &evaluator);
  result.saveTxt(path_join({base_dir, "filtering"}));

  std::cout << "\tsmoothing...\n";
  result =  fb.smoothing(data.obs, &evaluator);
  result.saveTxt(path_join({base_dir, "smoothing"}));

  std::cout << "\tonline smoothing...\n";
  result =  fb.online_smoothing(data.obs, lag, &evaluator);
  result.saveTxt(path_join({base_dir, "online_smoothing"}));

  std::cout << "done.\n\n"
            << "For visualization run command:\n\n"
            << "python ../visualize/test_comp.py "
            << base_dir << " "  << std::to_string(M) << std::endl;

};


int main(int argc, char *argv[]){

  std::string base_dir = "/tmp";
  if( argc > 1 )
    base_dir = argv[1];

  test_comp(base_dir);

  return 0;
}
