#include <bcpm_dm.hpp>
#include <bcpm_pg.hpp>
#include "bcpm_comp.hpp"

using namespace std;
using namespace pml;

void test_potential_multiplication(){

  std::cout << "Testing Potential Multiplication...\n";

  Vector alpha1 = {1, 2, 3, 4};
  Vector a1 = {10};
  Vector b1 = {1};

  Vector alpha2 = {5, 6, 7, 8};
  Vector a2 = {5};
  Vector b2 = {2};
  Vector empty = {};

  Vector alpha_result = {5,7,9,11};
  Vector a_result = {14};
  Vector b_result = {0.66666667};


  // Part 1: Multiply Dirichlets
  CompoundPotential cp1(alpha1, empty, empty);
  CompoundPotential cp2(alpha2, empty, empty);
  CompoundPotential* cp3 = (CompoundPotential*)(cp1 * cp2);
  assert(fequal(cp3->log_c, 2.62466487));
  assert(cp3->alpha.equals(alpha_result));
  delete cp3;

  // Part 2: Multiply Gammas
  cp1 = CompoundPotential(empty, a1, b1);
  cp2 = CompoundPotential(empty, a2, b2);
  cp3 = (CompoundPotential*)(cp1 * cp2);
  assert(fequal(cp3->log_c, -2.56996487));
  assert(cp3->a.equals(a_result));
  assert(cp3->b.equals(b_result));
  delete cp3;

  // Part 3: Multiply Both
  cp1 = CompoundPotential (alpha1, a1, b1);
  cp2 = CompoundPotential (alpha2, a2, b2);
  cp3 = (CompoundPotential*)(cp1 * cp2);
  assert(fequal(cp3->log_c, 2.62466487 - 2.56996487));
  assert(cp3->alpha.equals(alpha_result));
  assert(cp3->a.equals(a_result));
  assert(cp3->b.equals(b_result));
  delete cp3;

  std::cout << "done.\n";
}

int main(){
  test_potential_multiplication();
  return 0;
}
