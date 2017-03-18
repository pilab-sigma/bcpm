#include "bcpm_pg.hpp"

using namespace std;
using namespace pml;


void test_pg(const string &base_dir){
  cout << "Testing PG Model...\n";

  const double threshold = 0.99;
  const size_t window = 1;
  const double c = 0.01;
  const double a = 10;
  const double b = 1;
  const size_t length = 1000;

  // Generate model:
  PG_Model model(a, b, c);

  // Generate sequence
  auto data = model.generateData(length);
  data.saveTxt(path_join({base_dir, "data"}));

  // Estimate with true parameters
  ForwardBackward fb(&model);

  Evaluator evaluator(data.cps, threshold, window);

  std::cout << "\tfiltering...\n";
  auto result = fb.filtering(data.obs, &evaluator);
  result.saveTxt(path_join({base_dir, "filtering"}));

  std::cout << "\tsmoothing...\n";
  result = fb.smoothing(data.obs, &evaluator);
  result.saveTxt(path_join({base_dir, "smoothing"}));

  std::cout << "\tonline smoothing...\n";
  size_t lag = 10;
  result = fb.online_smoothing(data.obs, lag, &evaluator);
  result.saveTxt(path_join({base_dir, "online_smoothing"}));

  cout << "done.\n\n"
       << "For visualization run command:\n\n"
       << "python ../visualize/test_pg.py " + base_dir << std::endl;

}

int main(int argc, char *argv[]){

  std::string base_dir = "/tmp";
  if(argc == 2)
    base_dir = argv[1];

  test_pg(base_dir);

  return 0;
}