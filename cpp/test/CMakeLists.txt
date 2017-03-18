include_directories(../include)

link_libraries(gsl)
link_libraries(gslcblas)

add_executable(test_bcpm test_bcpm.cc)

add_executable(test_dm test_dm.cc)
add_executable(test_dm_em test_dm_em.cc)

add_executable(test_pg test_pg.cc)
add_executable(test_pg_em test_pg_em.cc)

add_executable(test_gaussian test_gaussian.cc)
add_executable(test_gaussian_em test_gaussian_em.cc)

add_executable(test_compound1 test_compound1.cc)
add_executable(test_compound2 test_compound2.cc)
add_executable(test_compound3 test_compound3.cc)

