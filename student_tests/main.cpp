// Tell Catch2 that it should not consider EXCEPTION_INT_DIVIDE_BY_ZERO to be a fatal exception.
// Compile with the /EHa flag to turn integer divide by zero into exceptions (instead of Windows killing the program immediately).
#define CATCH_CONFIG_NO_WINDOWS_SEH 1
#define CATCH_CONFIG_MAIN
#include "cpp_tests.h"
