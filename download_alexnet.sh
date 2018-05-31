#!/bin/bash
cd convert-models
pipenv install
# clone the caffe-tensorflow repository
git clone https://github.com/ethereon/caffe-tensorflow.git
# fix a bug in the repository
mv caffe-tensorflow/kaffe/caffe/caffepb.py caffe-tensorflow/kaffe/caffe/caffe_pb2.py
mv caffe-tensorflow/kaffe/caffe/resolver.py caffe-tensorflow/kaffe/caffe/resolver.bug
mv caffe-tensorflow/kaffe/tensorflow/network.py caffe-tensorflow/kaffe/tensorflow/network.bug
pipenv run python fix_bug.py
rm caffe-tensorflow/kaffe/caffe/resolver.bug
rm caffe-tensorflow/kaffe/tensorflow/network.bug
# download model
wget http://dl.caffe.berkeleyvision.org/bvlc_alexnet.caffemodel
wget https://raw.githubusercontent.com/BVLC/caffe/master/models/bvlc_alexnet/deploy.prototxt
# convert model to tensorflow format
pipenv run python caffe-tensorflow/convert.py --caffemodel=bvlc_alexnet.caffemodel deploy.prototxt --data-output-path=bvlc_alexnet.npy --code-output-path=bvlc_alexnet.py
# convert to pb dataformat
pipenv run python convert_to_pb.py
mv bvlc_alexnet.pb ../deep-spectrum/AlexNet.pb
rm bvlc_alexnet.*
rm deploy.prototxt
