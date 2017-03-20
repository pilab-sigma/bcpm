#include <bcpm_dm.hpp>
#include <bcpm_pg.hpp>
#include "bcpm_comp.hpp"

using namespace std;
using namespace pml;


void run_model(const string &base_dir, const string& method) {

  cout << "Testing compound model...\n";

  const std::string model_path = path_join({base_dir, "model.txt"});
  const std::string data_dir = path_join({base_dir, "data"});
  const std::string cpp_dir = path_join({base_dir, "cpp"});

  find_or_create(cpp_dir);

  COMP_Model model;
  model.loadTxt(model_path);
  const size_t lag = 10;

  // load data instead of generating a sequence
  ChangePointData data(data_dir);

  ForwardBackward fb(&model);
  if (method.empty() || method=="filtering") {
    std::cout << "\tfiltering...\n";
    auto result_filter = fb.filtering(data.obs);
    result_filter.saveTxt(path_join({cpp_dir, "filtering"}));
  }

  if (method.empty() || method=="smoothing") {
    std::cout << "\tsmoothing...\n";
    auto result_smooth = fb.smoothing(data.obs);
    result_smooth.saveTxt(path_join({cpp_dir, "smoothing"}));
  }

  if (method.empty() || method=="online_smoothing") {
    std::cout << "\tonline smoothing...\n";
    auto result_online_smooth = fb.online_smoothing(data.obs, lag);
    result_online_smooth.saveTxt(path_join({cpp_dir, "online_smoothing"}));
  }

};


int main(int argc, char *argv[]){

  std::string method = "";
  std::string base_dir = "/tmp/demo";
  if( argc > 1 )
    base_dir = argv[1];

  if( argc > 2 )
    method = argv[2];

  run_model(base_dir, method);

  return 0;
}

