% add path for caffe's matlab interface
addpath('/home/hhj/caffe-master/matlab');

kProtoFilePath = '/home/hhj/caffe-master/examples/mnist/lenet_train_test.prototxt';
kModelFilePath = '/home/hhj/caffe-master/examples/mnist/lenet_iter_10000.caffemodel';

% load caffe model
caffeNet = caffe.Net(kProtoFilePath, kModelFilePath, 'test');

% A = imread('/home/hhj/caffe-master/examples/mnist/testingdata/item1.bmp');
% width = 28;
% height = 28;
% [M,N] = size(A);
% im = zeros(M,N,3);
% im(:,:,1) = A;
% im(:,:,2) = A;
% im(:,:,3) = A;
% input_data = {prepare_image(im,width,height)};
% 
% scores = caffeNet.forward(input_data);
% [~, maxlabel] = max(scores);

% read parameters in the <conv_1> layer
% convKnlLst = caffeNet.layer_vec(kLayerIndConv).params(1).get_data();
% biasVecLst = caffeNet.layer_vec(kLayerIndConv).params(2).get_data();
% 
% % read parameters in the <fc_6> layer
% fcntWeiMat = caffeNet.layer_vec(kLayerIndFCnt).params(1).get_data();
% biasVecLst = caffeNet.layer_vec(kLayerIndFCnt).params(2).get_data();

% reset caffe model
caffe.reset_all();
