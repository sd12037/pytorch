	       7�q9	�5A�t�AX�Q #      ;�	��A�t�A"�$
E
#l3/bias(parameters)_140026356105288AccumulateGrad*
shape[2]
G
#l2/bias(parameters)_140026356371304AccumulateGrad*
shape[128]
G
#l1/bias(parameters)_140026356371128AccumulateGrad*
shape[512]
<
inputTensor_140026340021136AccumulateGrad*
shape[]
T
(conv1/weight(parameters)_140024832753072AccumulateGrad*
shape[64, 1, 2, 2]
�
ConvNd_140026340020968ConvNdBackwardinputTensor_140026340021136(conv1/weight(parameters)_140024832753072NoneType140026611571920*
shape[]
_
FeatureDropout_140024857799320FeatureDropoutBackwardConvNd_140026340020968*
shape[]
]
MaxPool2d_140024857799560MaxPool2dBackwardFeatureDropout_140024857799320*
shape[]
V
(conv2/weight(parameters)_140026356322376AccumulateGrad*
shape[128, 64, 2, 2]
�
ConvNd_140026340020576ConvNdBackwardMaxPool2d_140024857799560(conv2/weight(parameters)_140026356322376NoneType140026611571920*
shape[]
_
FeatureDropout_140024857799800FeatureDropoutBackwardConvNd_140026340020576*
shape[]
]
MaxPool2d_140024857800040MaxPool2dBackwardFeatureDropout_140024857799800*
shape[]
U
(conv3/weight(parameters)_140026356322464AccumulateGrad*
shape[1, 128, 2, 2]
�
ConvNd_140026340020296ConvNdBackwardMaxPool2d_140024857800040(conv3/weight(parameters)_140026356322464NoneType140026611571920*
shape[]
_
FeatureDropout_140024857800280FeatureDropoutBackwardConvNd_140026340020296*
shape[]
m
!AdaptiveMaxPool2d_140026356269128AdaptiveMaxPool2dBackwardFeatureDropout_140024857800280*
shape[]
\
Squeeze_140026356269368SqueezeBackward!AdaptiveMaxPool2d_140026356269128*
shape[]
S
-lstm/weight_ih_l0(parameters)_140026356370160AccumulateGrad*
shape	[40, 5]
T
-lstm/weight_hh_l0(parameters)_140026356370248AccumulateGrad*
shape
[40, 10]
N
+lstm/bias_ih_l0(parameters)_140026356370336AccumulateGrad*
shape[40]
N
+lstm/bias_hh_l0(parameters)_140026356370600AccumulateGrad*
shape[40]
[
5lstm/weight_ih_l0_reverse(parameters)_140026356370688AccumulateGrad*
shape	[40, 5]
\
5lstm/weight_hh_l0_reverse(parameters)_140026356370776AccumulateGrad*
shape
[40, 10]
V
3lstm/bias_ih_l0_reverse(parameters)_140026356370864AccumulateGrad*
shape[40]
V
3lstm/bias_hh_l0_reverse(parameters)_140026356370952AccumulateGrad*
shape[40]
�
CudnnRNN140026356269608CudnnRNNSqueeze_140026356269368-lstm/weight_ih_l0(parameters)_140026356370160-lstm/weight_hh_l0(parameters)_140026356370248+lstm/bias_ih_l0(parameters)_140026356370336+lstm/bias_hh_l0(parameters)_1400263563706005lstm/weight_ih_l0_reverse(parameters)_1400263563706885lstm/weight_hh_l0_reverse(parameters)_1400263563707763lstm/bias_ih_l0_reverse(parameters)_1400263563708643lstm/bias_hh_l0_reverse(parameters)_140026356370952NoneType140026611571920NoneType140026611571920*
shape[]
L
View_140026356269848ViewBackwardCudnnRNN140026356269608*
shape[]
N
%l1/weight(parameters)_140026356371040AccumulateGrad*
shape
[512, 100]
d
Transpose_140026356270088TransposeBackward%l1/weight(parameters)_140026356371040*
shape[]
�
Addmm_140026356270328AddmmBackward#l1/bias(parameters)_140026356371128View_140026356269848Transpose_140026356270088*
shape[]
T
Threshold_140026356270568ThresholdBackwardAddmm_140026356270328*
shape[]
T
Dropout_140026356270808DropoutBackwardThreshold_140026356270568*
shape[]
N
%l2/weight(parameters)_140026356371216AccumulateGrad*
shape
[128, 512]
d
Transpose_140026356271048TransposeBackward%l2/weight(parameters)_140026356371216*
shape[]
�
Addmm_140026356271288AddmmBackward#l2/bias(parameters)_140026356371304Dropout_140026356270808Transpose_140026356271048*
shape[]
K
'bnl2/weight(parameters)_140026356105552AccumulateGrad*
shape[128]
I
%bnl2/bias(parameters)_140026356105640AccumulateGrad*
shape[128]
�
BatchNorm_140026356063088BatchNormBackwardAddmm_140026356271288'bnl2/weight(parameters)_140026356105552%bnl2/bias(parameters)_140026356105640*
shape[]
X
Threshold_140026356271528ThresholdBackwardBatchNorm_140026356063088*
shape[]
T
Dropout_140026356271768DropoutBackwardThreshold_140026356271528*
shape[]
L
%l3/weight(parameters)_140026356371392AccumulateGrad*
shape
[2, 128]
d
Transpose_140026356272008TransposeBackward%l3/weight(parameters)_140026356371392*
shape[]
�
Addmm_140026356272248AddmmBackward#l3/bias(parameters)_140026356105288Dropout_140026356271768Transpose_140026356272008*
shape[]"e�tu#       ��wC	�;B�t�A*

training_loss��A��:'       ��F	z;B�t�A*

training_accuracy��?����