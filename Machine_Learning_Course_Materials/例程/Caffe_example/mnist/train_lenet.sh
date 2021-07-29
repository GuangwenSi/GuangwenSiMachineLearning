#!/usr/bin/env sh
set -e

BUILD=/home/hhj/caffe-master/build/tools
DATA=/home/hhj/caffe-master/examples/mnist
$BUILD/caffe train --solver=$DATA/lenet_solver.prototxt $@
