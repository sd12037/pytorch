	       7�q9	��Y�t�A;s�#      ;�	1��Y�t�A"�$
E
#l3/bias(parameters)_140621939974392AccumulateGrad*
shape[2]
G
#l2/bias(parameters)_140621939974216AccumulateGrad*
shape[128]
G
#l1/bias(parameters)_140621940240232AccumulateGrad*
shape[512]
<
inputTensor_140621934277520AccumulateGrad*
shape[]
T
(conv1/weight(parameters)_140620301790816AccumulateGrad*
shape[64, 1, 2, 2]
�
ConvNd_140621934277352ConvNdBackwardinputTensor_140621934277520(conv1/weight(parameters)_140620301790816NoneType140622080679120*
shape[]
_
FeatureDropout_140620326840984FeatureDropoutBackwardConvNd_140621934277352*
shape[]
]
MaxPool2d_140620326841224MaxPool2dBackwardFeatureDropout_140620326840984*
shape[]
V
(conv2/weight(parameters)_140621940191480AccumulateGrad*
shape[128, 64, 2, 2]
�
ConvNd_140621934276960ConvNdBackwardMaxPool2d_140620326841224(conv2/weight(parameters)_140621940191480NoneType140622080679120*
shape[]
_
FeatureDropout_140620326841464FeatureDropoutBackwardConvNd_140621934276960*
shape[]
]
MaxPool2d_140620326841704MaxPool2dBackwardFeatureDropout_140620326841464*
shape[]
U
(conv3/weight(parameters)_140621940191568AccumulateGrad*
shape[1, 128, 2, 2]
�
ConvNd_140621934276680ConvNdBackwardMaxPool2d_140620326841704(conv3/weight(parameters)_140621940191568NoneType140622080679120*
shape[]
_
FeatureDropout_140620326841944FeatureDropoutBackwardConvNd_140621934276680*
shape[]
m
!AdaptiveMaxPool2d_140621940138056AdaptiveMaxPool2dBackwardFeatureDropout_140620326841944*
shape[]
\
Squeeze_140621940138296SqueezeBackward!AdaptiveMaxPool2d_140621940138056*
shape[]
S
-lstm/weight_ih_l0(parameters)_140621940239264AccumulateGrad*
shape	[40, 5]
T
-lstm/weight_hh_l0(parameters)_140621940239352AccumulateGrad*
shape
[40, 10]
N
+lstm/bias_ih_l0(parameters)_140621940239440AccumulateGrad*
shape[40]
N
+lstm/bias_hh_l0(parameters)_140621940239704AccumulateGrad*
shape[40]
[
5lstm/weight_ih_l0_reverse(parameters)_140621940239792AccumulateGrad*
shape	[40, 5]
\
5lstm/weight_hh_l0_reverse(parameters)_140621940239880AccumulateGrad*
shape
[40, 10]
V
3lstm/bias_ih_l0_reverse(parameters)_140621940239968AccumulateGrad*
shape[40]
V
3lstm/bias_hh_l0_reverse(parameters)_140621940240056AccumulateGrad*
shape[40]
�
CudnnRNN140621940138536CudnnRNNSqueeze_140621940138296-lstm/weight_ih_l0(parameters)_140621940239264-lstm/weight_hh_l0(parameters)_140621940239352+lstm/bias_ih_l0(parameters)_140621940239440+lstm/bias_hh_l0(parameters)_1406219402397045lstm/weight_ih_l0_reverse(parameters)_1406219402397925lstm/weight_hh_l0_reverse(parameters)_1406219402398803lstm/bias_ih_l0_reverse(parameters)_1406219402399683lstm/bias_hh_l0_reverse(parameters)_140621940240056NoneType140622080679120NoneType140622080679120*
shape[]
L
View_140621940138776ViewBackwardCudnnRNN140621940138536*
shape[]
N
%l1/weight(parameters)_140621940240144AccumulateGrad*
shape
[512, 100]
d
Transpose_140621940139016TransposeBackward%l1/weight(parameters)_140621940240144*
shape[]
�
Addmm_140621940139256AddmmBackward#l1/bias(parameters)_140621940240232View_140621940138776Transpose_140621940139016*
shape[]
T
Threshold_140621940139496ThresholdBackwardAddmm_140621940139256*
shape[]
T
Dropout_140621940139736DropoutBackwardThreshold_140621940139496*
shape[]
N
%l2/weight(parameters)_140621940240320AccumulateGrad*
shape
[128, 512]
d
Transpose_140621940139976TransposeBackward%l2/weight(parameters)_140621940240320*
shape[]
�
Addmm_140621940140216AddmmBackward#l2/bias(parameters)_140621939974216Dropout_140621940139736Transpose_140621940139976*
shape[]
K
'bnl2/weight(parameters)_140621939974656AccumulateGrad*
shape[128]
I
%bnl2/bias(parameters)_140621939974744AccumulateGrad*
shape[128]
�
BatchNorm_140621939927920BatchNormBackwardAddmm_140621940140216'bnl2/weight(parameters)_140621939974656%bnl2/bias(parameters)_140621939974744*
shape[]
X
Threshold_140621940140456ThresholdBackwardBatchNorm_140621939927920*
shape[]
T
Dropout_140621940140696DropoutBackwardThreshold_140621940140456*
shape[]
L
%l3/weight(parameters)_140621939974304AccumulateGrad*
shape
[2, 128]
d
Transpose_140621940140936TransposeBackward%l3/weight(parameters)_140621939974304*
shape[]
�
Addmm_140621940141176AddmmBackward#l3/bias(parameters)_140621939974392Dropout_140621940140696Transpose_140621940140936*
shape[]" �Ֆ#       ��wC	�Z�t�A*

training_loss�A<�'       ��F	7	Z�t�A*

training_accuracyIr?��[6