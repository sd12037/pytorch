	       7�q9	r�.�t�A5�'K#      ;�	�+D/�t�A"�$
E
#l3/bias(parameters)_140111574458440AccumulateGrad*
shape[2]
G
#l2/bias(parameters)_140111574728552AccumulateGrad*
shape[128]
G
#l1/bias(parameters)_140111574728376AccumulateGrad*
shape[512]
<
inputTensor_140111558378384AccumulateGrad*
shape[]
T
(conv1/weight(parameters)_140110043520432AccumulateGrad*
shape[64, 1, 2, 2]
�
ConvNd_140111558378216ConvNdBackwardinputTensor_140111558378384(conv1/weight(parameters)_140110043520432NoneType140111822298320*
shape[]
_
FeatureDropout_140110068566680FeatureDropoutBackwardConvNd_140111558378216*
shape[]
]
MaxPool2d_140110068566920MaxPool2dBackwardFeatureDropout_140110068566680*
shape[]
V
(conv2/weight(parameters)_140111574679624AccumulateGrad*
shape[128, 64, 2, 2]
�
ConvNd_140111558377824ConvNdBackwardMaxPool2d_140110068566920(conv2/weight(parameters)_140111574679624NoneType140111822298320*
shape[]
_
FeatureDropout_140110068567160FeatureDropoutBackwardConvNd_140111558377824*
shape[]
]
MaxPool2d_140110068567400MaxPool2dBackwardFeatureDropout_140110068567160*
shape[]
U
(conv3/weight(parameters)_140111574679712AccumulateGrad*
shape[1, 128, 2, 2]
�
ConvNd_140111558377544ConvNdBackwardMaxPool2d_140110068567400(conv3/weight(parameters)_140111574679712NoneType140111822298320*
shape[]
_
FeatureDropout_140110068567640FeatureDropoutBackwardConvNd_140111558377544*
shape[]
m
!AdaptiveMaxPool2d_140111574630472AdaptiveMaxPool2dBackwardFeatureDropout_140110068567640*
shape[]
\
Squeeze_140111574630712SqueezeBackward!AdaptiveMaxPool2d_140111574630472*
shape[]
S
-lstm/weight_ih_l0(parameters)_140111574727408AccumulateGrad*
shape	[40, 5]
T
-lstm/weight_hh_l0(parameters)_140111574727496AccumulateGrad*
shape
[40, 10]
N
+lstm/bias_ih_l0(parameters)_140111574727584AccumulateGrad*
shape[40]
N
+lstm/bias_hh_l0(parameters)_140111574727848AccumulateGrad*
shape[40]
[
5lstm/weight_ih_l0_reverse(parameters)_140111574727936AccumulateGrad*
shape	[40, 5]
\
5lstm/weight_hh_l0_reverse(parameters)_140111574728024AccumulateGrad*
shape
[40, 10]
V
3lstm/bias_ih_l0_reverse(parameters)_140111574728112AccumulateGrad*
shape[40]
V
3lstm/bias_hh_l0_reverse(parameters)_140111574728200AccumulateGrad*
shape[40]
�
CudnnRNN140111574630952CudnnRNNSqueeze_140111574630712-lstm/weight_ih_l0(parameters)_140111574727408-lstm/weight_hh_l0(parameters)_140111574727496+lstm/bias_ih_l0(parameters)_140111574727584+lstm/bias_hh_l0(parameters)_1401115747278485lstm/weight_ih_l0_reverse(parameters)_1401115747279365lstm/weight_hh_l0_reverse(parameters)_1401115747280243lstm/bias_ih_l0_reverse(parameters)_1401115747281123lstm/bias_hh_l0_reverse(parameters)_140111574728200NoneType140111822298320NoneType140111822298320*
shape[]
L
View_140111574631192ViewBackwardCudnnRNN140111574630952*
shape[]
N
%l1/weight(parameters)_140111574728288AccumulateGrad*
shape
[512, 100]
d
Transpose_140111574631432TransposeBackward%l1/weight(parameters)_140111574728288*
shape[]
�
Addmm_140111574631672AddmmBackward#l1/bias(parameters)_140111574728376View_140111574631192Transpose_140111574631432*
shape[]
T
Threshold_140111574631912ThresholdBackwardAddmm_140111574631672*
shape[]
T
Dropout_140111574632152DropoutBackwardThreshold_140111574631912*
shape[]
N
%l2/weight(parameters)_140111574728464AccumulateGrad*
shape
[128, 512]
d
Transpose_140111574632392TransposeBackward%l2/weight(parameters)_140111574728464*
shape[]
�
Addmm_140111574632632AddmmBackward#l2/bias(parameters)_140111574728552Dropout_140111574632152Transpose_140111574632392*
shape[]
K
'bnl2/weight(parameters)_140111574458704AccumulateGrad*
shape[128]
I
%bnl2/bias(parameters)_140111574458792AccumulateGrad*
shape[128]
�
BatchNorm_140111574416240BatchNormBackwardAddmm_140111574632632'bnl2/weight(parameters)_140111574458704%bnl2/bias(parameters)_140111574458792*
shape[]
X
Threshold_140111574632872ThresholdBackwardBatchNorm_140111574416240*
shape[]
T
Dropout_140111574633112DropoutBackwardThreshold_140111574632872*
shape[]
L
%l3/weight(parameters)_140111574728640AccumulateGrad*
shape
[2, 128]
d
Transpose_140111574633352TransposeBackward%l3/weight(parameters)_140111574728640*
shape[]
�
Addmm_140111574633592AddmmBackward#l3/bias(parameters)_140111574458440Dropout_140111574633112Transpose_140111574633352*
shape[]"���