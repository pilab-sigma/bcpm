#include "bcpm.hpp"

using namespace std;
using namespace pml;


const double threshold = 0.99;
const size_t window = 1;

void test_dm(){
  cout << "test_dm()...\n";

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
  Matrix mean;
  Vector cpp;
  DM_ForwardBackward fb(&model);

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
}

void test_dm_em(){

  cout << "test_dm_em()...\n";

  size_t K = 5;
  double precision = K;
  bool fixed_precision = true;
  Vector alpha = normalize(Vector::ones(K)) * precision;
  double p1 = 0.1;

  size_t length = 100;

  // Generate model:
  DM_Model model(alpha, p1, fixed_precision);

  // Generate sequence
  auto data = model.generateData(length);
  data.saveTxt("/tmp");

  Evaluator evaluator(data.cps, threshold, window);

  // Estimate with true parameters
  DM_ForwardBackward fb(&model);
  auto result = fb.smoothing(data.obs, &evaluator);
  result.saveTxt("/tmp");

  // Learn parameters
  double c_init = 0.0001;
  DM_Model em_model(DirichletPotential::rand_gen(K, precision).alpha,
                    c_init, fixed_precision);
  DM_Model em_init_model = em_model;
  DM_ForwardBackward fb_em(&em_model);

  // Run with EM inital
  result = fb_em.smoothing(data.obs, &evaluator);
  result.saveTxt("/tmp/initial");

  // Learn parameters
  result = fb_em.learn_parameters(data.obs, &evaluator);
  result.saveTxt("/tmp/final");

  std::cout << "-----------\n";
  std::cout << "True model:\n";
  model.print();
  std::cout << "-----------\n";
  std::cout << "EM(initial) model:\n";
  em_init_model.print();
  std::cout << "-----------\n";
  std::cout << "EM(final) model:\n";
  em_model.print();
  std::cout << "-----------\n";

  if(system("anaconda3 ../test/python/test_bcpm_dm.py True")){
    std::cout <<"plotting error...\n";
  }

  cout << "OK.\n";
}


void test_pg(){

  cout << "test_pg()...\n";
  size_t length = 1000;
  double c = 0.01;
  double a = 10;
  double b = 1;

  // Generate model:
  PG_Model model(a, b, c);

  // Generate sequence
  auto data = model.generateData(length);
  data.saveTxt("/tmp");

  // Estimate with true parameters
  PG_ForwardBackward fb(&model);

  Evaluator evaluator(data.cps, threshold, window);

  std::cout << "Filtering...\n";
  auto result = fb.filtering(data.obs, &evaluator);
  result.saveTxt("/tmp/filtering");

  std::cout << "Smoothing...\n";
  result = fb.smoothing(data.obs, &evaluator);
  result.saveTxt("/tmp/smoothing");


  std::cout << "Online smoothing...\n";
  size_t lag = 10;
  result = fb.online_smoothing(data.obs, lag, &evaluator);
  result.saveTxt("/tmp/online_smoothing");

  if(system("anaconda3 ../test/python/test_bcpm_pg.py False")){
    std::cout <<"plotting error...\n";
  }
  cout << "OK.\n";
}

void test_pg_em(){
  cout << "test_pg_em...\n";
  size_t length = 200;
  double p1 = 0.05;
  double a = 10;
  double b = 1;
  bool fixed_scale = true;

  // Generate model:
  PG_Model model(a, b, p1, fixed_scale);

  // Generate Sequence
  auto data = model.generateData(length);
  data.saveTxt("/tmp");

  Evaluator evaluator(data.cps, threshold, window);

  // Smoothing with true parameters
  PG_ForwardBackward fb(&model);
  Result result = fb.smoothing(data.obs, &evaluator);
  result.saveTxt("/tmp");

  // Generate random model for EM
  double init_p1 = 0.001;
  double init_a = Uniform(0, 10).rand();
  double init_b = 1;

  PG_Model init_model(init_a, init_b, init_p1, fixed_scale);
  PG_Model em_model = init_model;
  PG_ForwardBackward fb_em(&em_model);

  // Run initial model:
  result = fb_em.smoothing(data.obs, &evaluator);
  result.saveTxt("/tmp/initial");

  // Run EM:
  result = fb_em.learn_parameters(data.obs, &evaluator);
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

  if(system("anaconda3 ../test/python/test_bcpm_pg.py True")){
    std::cout <<"plotting error...\n";
  }
  cout << "OK.\n";
}

void test_g(){

  double p1 = 0.01;
  double mu = 3;
  double sigma = 2;
  size_t length = 200;
  size_t lag = 10;

  // Generate model:
  G_Model model(mu, sigma, p1);

  // Generate Sequence
  auto data = model.generateData(length);
  data.saveTxt("/tmp");

  // Estimate with true parameters
  G_ForwardBackward fb(&model);

  std::cout << "Filtering...\n";
  auto result = fb.filtering(data.obs);
  result.saveTxt("/tmp/filtering");

  std::cout << "Smoothing...\n";
  result = fb.smoothing(data.obs);
  result.saveTxt("/tmp/smoothing");

  std::cout << "Online smoothing...\n";
  result = fb.online_smoothing(data.obs, lag);
  result.saveTxt("/tmp/online_smoothing");

  std::cout << "Visualizing...\n";
  if(system("anaconda3 ../test/python/test_bcpm_pg.py False")){
    std::cout <<"plotting error...\n";
  }
  cout << "OK.\n";

}

void test_g_em(){

  double p1 = 0.01;
  double mu = 3;
  double sigma = 2;
  size_t length = 200;

  // Generate data:
  G_Model model(mu, sigma, p1);

  // Generate Sequence
  auto data = model.generateData(length);
  data.saveTxt("/tmp");

  // Estimate with true parameters
  G_ForwardBackward fb(&model);
  auto result = fb.smoothing(data.obs);
  result.saveTxt("/tmp");


  // Random init a model
  G_Model init_model(Uniform(0, 5).rand(), Uniform(0, 5).rand(), p1);
  G_Model em_model = init_model;

  G_ForwardBackward fb_em(&em_model);
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

  std::cout << "Visualizing...\n";
  if(system("anaconda3 ../test/python/test_bcpm_pg.py True"))
    std::cout <<"plotting error...\n";
  cout << "OK.\n";

}

int main() {

  //test_dm();
  //test_dm_em();

  test_pg();
  //test_pg_em();

  //test_g();
  //test_g_em();

  return 0;
}
