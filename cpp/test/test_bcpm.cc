#include <cassert>

#include "bcpm_dm.hpp"

using namespace std;
using namespace pml;

void test_potential() {
  std::cout << "test_potential...\n";

  Vector alpha = {1,2,3,4,5};
  double log_c = -0.1;

  DirichletPotential *dm = new DirichletPotential(alpha, log_c);
  assert(alpha.equals(dm->alpha));
  assert(fequal(log_c, dm->log_c));

  DirichletPotential *dm2  = new DirichletPotential(*dm);
  assert(dm->alpha.equals(dm2->alpha));
  assert(fequal(dm->log_c, dm2->log_c));
  assert( dm != dm2 );

  DirichletPotential *dm3  = (DirichletPotential *) dm2->clone();
  dm3->log_c += -0.1;
  assert(dm2->alpha.equals(dm3->alpha));
  assert(fequal(dm2->log_c - 0.1 , dm3->log_c));
  assert( dm2 != dm3 );

  delete dm;
  delete dm2;
  delete dm3;
  std::cout << "OK.\n";
}

void test_message() {
  std::cout << "test_message...\n";

  Message m;
  m.add_potential( new DirichletPotential(Uniform().rand(5)) );
  m.add_potential( new DirichletPotential(Uniform().rand(5)) );

  // Test Copy Constructor
  Message m2(m);
  DirichletPotential *d1 = (DirichletPotential*) m.potentials[0];
  DirichletPotential *d2 = (DirichletPotential*) m2.potentials[0];
  assert(d1->alpha.equals(d2->alpha));
  assert(d1 != d2);

  // Test Assignment
  Message m3;
  m3 = m;
  DirichletPotential *d3 = (DirichletPotential*) m3.potentials[0];
  assert(d1->alpha.equals(d2->alpha));
  assert(d1 != d3);

  // Test Move Copy
  Message m4(std::move(m));
  assert(m4.potentials.size() == 2);
  assert(m.potentials.size() == 0);
  DirichletPotential *d4 = (DirichletPotential*) m4.potentials[0];
  assert(d2->alpha.equals(d4->alpha));
  assert(d1 == d4);


  // Test Move Assignment
  Message m5;
  m5 = std::move(m4);
  assert(m5.potentials.size() == 2);
  assert(m4.potentials.size() == 0);
  DirichletPotential *d5 = (DirichletPotential*) m5.potentials[0];
  assert(d2->alpha.equals(d4->alpha));
  assert(d1 == d5);

  // Test Vector Push
  std::vector<Message> v;
  v.push_back(m5);
  DirichletPotential *dv = (DirichletPotential*) v[0].potentials[0];
  assert(dv->alpha.equals(d5->alpha));
  assert(dv != d5);

  // Test Vector Move Push
  v.clear();
  v.push_back(std::move(m5));
  dv = (DirichletPotential*) v[0].potentials[0];
  assert(m5.potentials.size() == 0);
  assert(dv == d5);

  std::cout << "OK.\n";
}

void test_message_prune(){

  std::cout << "test_message_prune...\n";

  Message m(3);
  DirichletPotential *p1 = new DirichletPotential(Uniform().rand(5), 1);
  DirichletPotential *p2 = new DirichletPotential(Uniform().rand(5), 1);
  DirichletPotential *p3 = new DirichletPotential(Uniform().rand(5), 2);
  DirichletPotential *p4 = new DirichletPotential(Uniform().rand(5), 3);

  m.add_potential( p1 );
  m.add_potential( p2 );
  m.add_potential( p3 );
  m.add_potential( p4 );

  assert(m.size() == 3);
  assert(m.potentials[0] == p1);
  assert(m.potentials[1] == p3);
  assert(m.potentials[2] == p4);

  std::cout << "OK.\n";
}


int main(){
  test_potential();
  test_message();
  test_message_prune();
  return 0;
}