include_directories(../include)

link_libraries(gsl)
link_libraries(gslcblas)

add_executable(test_dm test_dm.cc)
add_executable(test_dm_em test_dm_em.cc)

add_executable(test_pg test_pg.cc)
add_executable(test_pg_em test_pg_em.cc)


