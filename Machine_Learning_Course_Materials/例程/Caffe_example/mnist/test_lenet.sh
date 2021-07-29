#!/usr/bin/env sh
set -e

BUILD=/home/hhj/caffe-master/build/tools
DATA=/home/hhj/caffe-master/examples/mnist
$BUILD/caffe test -model $DATA/lenet_train_test.prototxt -weights $DATA/lenet_iter_10000.caffemodel -iterations 100 $@
