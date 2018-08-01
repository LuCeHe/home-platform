#!/bin/bash

DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )

export PYTHONPATH=$DIR/../:$PYTHONPATH

usage()
{
    echo "usage: ./run_tests.sh [[-c] | [-py3] | [-h]]"
}

coverage=0
python_version=2
while [ "$1" != "" ]; do
    case $1 in
        -c | --coverage )           coverage=1
                                    ;;
        -py3 | --python-version-3 ) python_version=3
                                    ;;
        -h | --help )               usage
                                    exit
                                    ;;
        * )                         usage
                                    exit 1
    esac
    shift
done

PYV=`python -c "import sys;t='{v[0]}.{v[1]}'.format(v=list(sys.version_info[:2]));sys.stdout.write(t)";`
# if the default Python is v3.x then we force using the -py3 switch
if (( $(echo "$PYV > 3.0" | bc -l) )); then
    python_version=3
fi

# Convert SUNCG test data
$DIR/../scripts/convert_suncg.sh "$DIR/data/suncg"

# Create output directories
mkdir -p ${DIR}/report/profile
mkdir -p ${DIR}/report/coverage

# Cleanup previous reports
target=${DIR}/report/coverage/
if find "$target" -mindepth 1 -print -quit | grep -q .; then
    # Output folder not empty, erase all existing files
    rm ${DIR}/report/coverage/*
fi

# Run unit tests and coverage analysis for the 'action' module
coverage_params=
packages=
if [[ coverage -eq 1 ]]; then
    packages="${packages}home_platform,"
    packages="${packages}home_platform.acoustics,"
    packages="${packages}home_platform.core,"
    packages="${packages}home_platform.physics,"
    packages="${packages}home_platform.rendering,"
    packages="${packages}home_platform.suncg,"
    packages="${packages}home_platform.utils,"
    packages="${packages}home_platform.env,"
    packages="${packages}home_platform.constants,"
    packages="${packages}home_platform.semantic"
    coverage_params="--with-coverage --cover-html --cover-html-dir=${DIR}/report/coverage
                     --cover-erase --cover-tests --cover-package=${packages}"
fi

nose_cmd="nosetests"
if [[ $python_version -eq 3 ]]; then
    nose_cmd="nosetests3"
fi

nose_call="${nose_cmd} --verbosity 3 ${coverage_params}"

echo ${nose_call}
eval ${nose_call}

if [[ coverage -eq 1 ]]; then
	x-www-browser ${DIR}/report/coverage/index.html
fi

# For profiler sorting options, see:
# https://docs.python.org/2/library/profile.html#pstats.Stats

#PROFILE_FILE=${DIR}/report/profile/profile.out
#python -m cProfile -o ${PROFILE_FILE} `which nosetests` ${DIR}
#runsnake ${PROFILE_FILE}
