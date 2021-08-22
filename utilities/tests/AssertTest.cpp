#define BOOST_TEST_MODULE AssertionTests
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MAIN

#include "../include/utilities/runtime/NeonAssert.h"
#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_CASE(TestNeonWarningAssertion) { NEON_ASSERT_WARN(false, "Is", "it", "working?"); }