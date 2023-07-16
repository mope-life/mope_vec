add_executable(tests tests.cxx)
target_include_directories(tests PUBLIC ..)
target_link_libraries(tests PRIVATE Catch2::Catch2WithMain)
catch_discover_tests(tests)
