	       7�q9	��|F�t�A�{�n#      ;�	�&;G�t�A"�$
E
#l3/bias(parameters)_140702652014664AccumulateGrad*
shape[2]
G
#l2/bias(parameters)_140702652280680AccumulateGrad*
shape[128]
G
#l1/bias(parameters)_140702652280504AccumulateGrad*
shape[512]
<
inputTensor_140702635934608AccumulateGrad*
shape[]
T
(conv1/weight(parameters)_140701124103600AccumulateGrad*
shape[64, 1, 2, 2]
�
ConvNd_140702635934440ConvNdBackwardinputTensor_140702635934608(conv1/weight(parameters)_140701124103600NoneType140702902910160*
shape[]
_
FeatureDropout_140701149149848FeatureDropoutBackwardConvNd_140702635934440*
shape[]
]
MaxPool2d_140701149150088MaxPool2dBackwardFeatureDropout_140701149149848*
shape[]
V
(conv2/weight(parameters)_140702652231752AccumulateGrad*
shape[128, 64, 2, 2]
�
ConvNd_140702635934048ConvNdBackwardMaxPool2d_140701149150088(conv2/weight(parameters)_140702652231752NoneType140702902910160*
shape[]
_
FeatureDropout_140701149150328FeatureDropoutBackwardConvNd_140702635934048*
shape[]
]
MaxPool2d_140701149150568MaxPool2dBackwardFeatureDropout_140701149150328*
shape[]
U
(conv3/weight(parameters)_140702652231840AccumulateGrad*
shape[1, 128, 2, 2]
�
ConvNd_140702635933768ConvNdBackwardMaxPool2d_140701149150568(conv3/weight(parameters)_140702652231840NoneType140702902910160*
shape[]
_
FeatureDropout_140701149150808FeatureDropoutBackwardConvNd_140702635933768*
shape[]
m
!AdaptiveMaxPool2d_140702652182600AdaptiveMaxPool2dBackwardFeatureDropout_140701149150808*
shape[]
\
Squeeze_140702652182840SqueezeBackward!AdaptiveMaxPool2d_140702652182600*
shape[]
S
-lstm/weight_ih_l0(parameters)_140702652279536AccumulateGrad*
shape	[40, 5]
T
-lstm/weight_hh_l0(parameters)_140702652279624AccumulateGrad*
shape
[40, 10]
N
+lstm/bias_ih_l0(parameters)_140702652279712AccumulateGrad*
shape[40]
N
+lstm/bias_hh_l0(parameters)_140702652279976AccumulateGrad*
shape[40]
[
5lstm/weight_ih_l0_reverse(parameters)_140702652280064AccumulateGrad*
shape	[40, 5]
\
5lstm/weight_hh_l0_reverse(parameters)_140702652280152AccumulateGrad*
shape
[40, 10]
V
3lstm/bias_ih_l0_reverse(parameters)_140702652280240AccumulateGrad*
shape[40]
V
3lstm/bias_hh_l0_reverse(parameters)_140702652280328AccumulateGrad*
shape[40]
�
CudnnRNN140702652183080CudnnRNNSqueeze_140702652182840-lstm/weight_ih_l0(parameters)_140702652279536-lstm/weight_hh_l0(parameters)_140702652279624+lstm/bias_ih_l0(parameters)_140702652279712+lstm/bias_hh_l0(parameters)_1407026522799765lstm/weight_ih_l0_reverse(parameters)_1407026522800645lstm/weight_hh_l0_reverse(parameters)_1407026522801523lstm/bias_ih_l0_reverse(parameters)_1407026522802403lstm/bias_hh_l0_reverse(parameters)_140702652280328NoneType140702902910160NoneType140702902910160*
shape[]
L
View_140702652183320ViewBackwardCudnnRNN140702652183080*
shape[]
N
%l1/weight(parameters)_140702652280416AccumulateGrad*
shape
[512, 100]
d
Transpose_140702652183560TransposeBackward%l1/weight(parameters)_140702652280416*
shape[]
�
Addmm_140702652183800AddmmBackward#l1/bias(parameters)_140702652280504View_140702652183320Transpose_140702652183560*
shape[]
T
Threshold_140702652184040ThresholdBackwardAddmm_140702652183800*
shape[]
T
Dropout_140702652184280DropoutBackwardThreshold_140702652184040*
shape[]
N
%l2/weight(parameters)_140702652280592AccumulateGrad*
shape
[128, 512]
d
Transpose_140702652184520TransposeBackward%l2/weight(parameters)_140702652280592*
shape[]
�
Addmm_140702652184760AddmmBackward#l2/bias(parameters)_140702652280680Dropout_140702652184280Transpose_140702652184520*
shape[]
K
'bnl2/weight(parameters)_140702652014928AccumulateGrad*
shape[128]
I
%bnl2/bias(parameters)_140702652015016AccumulateGrad*
shape[128]
�
BatchNorm_140702651972464BatchNormBackwardAddmm_140702652184760'bnl2/weight(parameters)_140702652014928%bnl2/bias(parameters)_140702652015016*
shape[]
X
Threshold_140702652185000ThresholdBackwardBatchNorm_140702651972464*
shape[]
T
Dropout_140702652185240DropoutBackwardThreshold_140702652185000*
shape[]
L
%l3/weight(parameters)_140702652280768AccumulateGrad*
shape
[2, 128]
d
Transpose_140702652185480TransposeBackward%l3/weight(parameters)_140702652280768*
shape[]
�
Addmm_140702652185720AddmmBackward#l3/bias(parameters)_140702652014664Dropout_140702652185240Transpose_140702652185480*
shape[]"�50�#       ��wC	^�G�t�A*

training_loss�� A�<{2'       ��F	l_�G�t�A*

training_accuracyn[?�o�