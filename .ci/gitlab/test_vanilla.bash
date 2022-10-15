#!/bin/bash

THIS_DIR="$(cd "$(dirname ${BASH_SOURCE[0]})" ; pwd -P )"
source ${THIS_DIR}/common_test_setup.bash
# this runs in pytest in a fake, auto numbered, X Server
#xvfb-run -a pytest ${COMMON_PYTEST_OPTS} --hypothesis-show-statistics
xvfb-run -a pytest ${COMMON_PYTEST_OPTS}
_coverage_xml
