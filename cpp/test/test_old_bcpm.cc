//
// Created by cagatay on 11.03.2017.
//


#include "../old/bcpm_compound.hpp"


using namespace std;
using namespace pml;


const double threshold = 0.99;
const size_t window = 1;

size_t length = 100;
double c = 0.01;

double alpha_ = 5;
double a_ = 10;
double b_ = 5;

size_t M = 5;
size_t N = 3;

int main() {
  Matrix data = Matrix::loadTxt("/tmp/obs.txt");

  Vector alpha = Vector::ones(M)*alpha_;
  Vector a = Vector::ones(N)*a_;
  Vector b = Vector::ones(N)*b_;

  Model model(c, Potential(alpha, a, b));

  Result result = model.filtering(data);
  cout << result.cpp << endl << sum(result.cpp) << endl;
}