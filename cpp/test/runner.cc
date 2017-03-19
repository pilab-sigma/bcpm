#include <bcpm_dm.hpp>
#include <bcpm_pg.hpp>
#include "bcpm_comp.hpp"

using namespace std;
using namespace pml;


void run_model(const string &base_dir, const string& method) {

  cout << "Testing compound model...\n";

  COMP_Model model;
  model.loadTxt(path_join({base_dir, "model.txt"}));
  const size_t lag = 10;

  // Generate Sequence
  // auto data = model.generateData(length);
  // data.saveTxt(path_join({base_dir, "data"}));

  // load data instead of generating a sequence
  // ChangePointData data(path_join({base_dir, "data"}));

  // load data v2
  ChangePointData data;
  data.obs = Matrix::loadTxt(path_join({base_dir, "data.txt"}));
  cout << "data.shape:\t" << data.obs.nrows() << ", " << data.obs.ncols() << endl;

  ForwardBackward fb(&model);
  if (method=="filtering") {
    std::cout << "\tfiltering...\n";
    auto result_filter = fb.filtering(data.obs);
    result_filter.saveTxt(path_join({base_dir, "filtering_cpp"}));
  }
  else if (method=="smoothing") {
    std::cout << "\tsmoothing...\n";
    auto result_smooth = fb.smoothing(data.obs);
    result_smooth.saveTxt(path_join({base_dir, "smoothing_cpp"}));
  }
  else if (method=="online_smoothing") {
    std::cout << "\tonline smoothing...\n";
    auto result_online_smooth = fb.online_smoothing(data.obs, lag);
    result_online_smooth.saveTxt(path_join({base_dir, "online_smoothing_cpp"}));
  }

};


int main(int argc, char *argv[]){

  std::string base_dir = "/tmp/demo";
  if( argc > 1 )
    base_dir = argv[1];

  std::string method= "online_smoothing";

  run_model(base_dir, method);

  return 0;
}
