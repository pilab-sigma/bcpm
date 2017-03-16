include_directories(../include)

link_libraries(gsl)
link_libraries(gslcblas)

add_executable(test_bcpm test_bcpm.cc)

add_executable(test_dm test_dm.cc)
add_executable(test_dm_em test_dm_em.cc)

add_executable(test_pg test_pg.cc)
add_executable(test_pg_em test_pg_em.cc)

add_executable(test_old_comp ../old/test_comp.cc)

add_executable(test_comp test_comp.cc)
add_executable(test_comp2 test_comp2.cc)

add_executable(test_old_bcpm test_old_bcpm.cc)

