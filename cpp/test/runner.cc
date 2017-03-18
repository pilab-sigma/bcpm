#include <bcpm_dm.hpp>
#include <bcpm_pg.hpp>
#include "bcpm_comp.hpp"

using namespace std;
using namespace pml;

const size_t M = 5;
const size_t N = 3;


void test_comp(const string &base_dir) {

  /*
  const double threshold = 0.99;
  const size_t window = 1;
  const size_t length = 100;

  const double c = 0.01;
  const double alpha_ = 5;
  const double a_ = 10;
  const double b_ = 5;

  Vector alpha = Vector::ones(M)*alpha_;
  Vector a = Vector::ones(N)*a_;
  Vector b = Vector::ones(N)*b_;
  // Generate model:
  COMP_Model model(alpha, a, b, c);
  */

  cout << "Testing compound model...\n";

  COMP_Model model(Vector::zeros(0),Vector::zeros(0),Vector::zeros(0),1);
  model.loadTxt(path_join({base_dir, "model.txt"}));
  const size_t lag = 10;

  // Generate Sequence
  // auto data = model.generateData(length);
  // data.saveTxt(path_join({base_dir, "data"}));

  ChangePointData data(path_join({base_dir, "data"}));

  ForwardBackward fb(&model);
  std::cout << "\tfiltering...\n";
  auto result =  fb.filtering(data.obs);
  result.saveTxt(path_join({base_dir, "filtering_cpp"}));

  std::cout << "\tsmoothing...\n";
  result =  fb.smoothing(data.obs);
  result.saveTxt(path_join({base_dir, "smoothing_cpp"}));

  std::cout << "\tonline smoothing...\n";
  result =  fb.online_smoothing(data.obs, lag);
  result.saveTxt(path_join({base_dir, "online_smoothing_cpp"}));

  std::cout << "done.\n\n"
  << "For visualization run command:\n\n"
  << "python ../visualize/test_comp.py "
  << base_dir << " "  << std::to_string(M) << std::endl;

};


int main(int argc, char *argv[]){

  std::string base_dir = "/tmp/demo";
  if( argc > 1 )
    base_dir = argv[1];

  test_comp(base_dir);

  return 0;
}
