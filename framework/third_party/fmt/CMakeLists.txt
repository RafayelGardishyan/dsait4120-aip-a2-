add_library(fmt "src/format.cc" "src/os.cc")
target_include_directories(fmt PUBLIC "include")
target_compile_definitions(fmt PUBLIC "FMT_UNICODE=0")
target_compile_features(fmt PUBLIC cxx_std_20)