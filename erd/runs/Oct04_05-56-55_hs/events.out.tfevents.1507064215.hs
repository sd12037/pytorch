	       7�q9	���e�t�A���I#      ;�	iA�f�t�A"�$
E
#l3/bias(parameters)_139765161132104AccumulateGrad*
shape[2]
G
#l2/bias(parameters)_139765161398120AccumulateGrad*
shape[128]
G
#l1/bias(parameters)_139765161397944AccumulateGrad*
shape[512]
<
inputTensor_139765145043800AccumulateGrad*
shape[]
T
(conv1/weight(parameters)_139763630972336AccumulateGrad*
shape[64, 1, 2, 2]
�
ConvNd_139765145043632ConvNdBackwardinputTensor_139765145043800(conv1/weight(parameters)_139763630972336NoneType139765409774800*
shape[]
_
FeatureDropout_139763656018584FeatureDropoutBackwardConvNd_139765145043632*
shape[]
]
MaxPool2d_139763656018824MaxPool2dBackwardFeatureDropout_139763656018584*
shape[]
V
(conv2/weight(parameters)_139765161349192AccumulateGrad*
shape[128, 64, 2, 2]
�
ConvNd_139765145043240ConvNdBackwardMaxPool2d_139763656018824(conv2/weight(parameters)_139765161349192NoneType139765409774800*
shape[]
_
FeatureDropout_139763656019064FeatureDropoutBackwardConvNd_139765145043240*
shape[]
]
MaxPool2d_139763656019304MaxPool2dBackwardFeatureDropout_139763656019064*
shape[]
U
(conv3/weight(parameters)_139765161349280AccumulateGrad*
shape[1, 128, 2, 2]
�
ConvNd_139765161086928ConvNdBackwardMaxPool2d_139763656019304(conv3/weight(parameters)_139765161349280NoneType139765409774800*
shape[]
_
FeatureDropout_139763656019544FeatureDropoutBackwardConvNd_139765161086928*
shape[]
m
!AdaptiveMaxPool2d_139765161295944AdaptiveMaxPool2dBackwardFeatureDropout_139763656019544*
shape[]
\
Squeeze_139765161296184SqueezeBackward!AdaptiveMaxPool2d_139765161295944*
shape[]
S
-lstm/weight_ih_l0(parameters)_139765161396976AccumulateGrad*
shape	[40, 5]
T
-lstm/weight_hh_l0(parameters)_139765161397064AccumulateGrad*
shape
[40, 10]
N
+lstm/bias_ih_l0(parameters)_139765161397152AccumulateGrad*
shape[40]
N
+lstm/bias_hh_l0(parameters)_139765161397416AccumulateGrad*
shape[40]
[
5lstm/weight_ih_l0_reverse(parameters)_139765161397504AccumulateGrad*
shape	[40, 5]
\
5lstm/weight_hh_l0_reverse(parameters)_139765161397592AccumulateGrad*
shape
[40, 10]
V
3lstm/bias_ih_l0_reverse(parameters)_139765161397680AccumulateGrad*
shape[40]
V
3lstm/bias_hh_l0_reverse(parameters)_139765161397768AccumulateGrad*
shape[40]
�
CudnnRNN139765161296424CudnnRNNSqueeze_139765161296184-lstm/weight_ih_l0(parameters)_139765161396976-lstm/weight_hh_l0(parameters)_139765161397064+lstm/bias_ih_l0(parameters)_139765161397152+lstm/bias_hh_l0(parameters)_1397651613974165lstm/weight_ih_l0_reverse(parameters)_1397651613975045lstm/weight_hh_l0_reverse(parameters)_1397651613975923lstm/bias_ih_l0_reverse(parameters)_1397651613976803lstm/bias_hh_l0_reverse(parameters)_139765161397768NoneType139765409774800NoneType139765409774800*
shape[]
L
View_139765161296664ViewBackwardCudnnRNN139765161296424*
shape[]
N
%l1/weight(parameters)_139765161397856AccumulateGrad*
shape
[512, 100]
d
Transpose_139765161296904TransposeBackward%l1/weight(parameters)_139765161397856*
shape[]
�
Addmm_139765161297144AddmmBackward#l1/bias(parameters)_139765161397944View_139765161296664Transpose_139765161296904*
shape[]
T
Threshold_139765161297384ThresholdBackwardAddmm_139765161297144*
shape[]
T
Dropout_139765161297624DropoutBackwardThreshold_139765161297384*
shape[]
N
%l2/weight(parameters)_139765161398032AccumulateGrad*
shape
[128, 512]
d
Transpose_139765161297864TransposeBackward%l2/weight(parameters)_139765161398032*
shape[]
�
Addmm_139765161298104AddmmBackward#l2/bias(parameters)_139765161398120Dropout_139765161297624Transpose_139765161297864*
shape[]
K
'bnl2/weight(parameters)_139765161132368AccumulateGrad*
shape[128]
I
%bnl2/bias(parameters)_139765161132456AccumulateGrad*
shape[128]
�
BatchNorm_139765161085752BatchNormBackwardAddmm_139765161298104'bnl2/weight(parameters)_139765161132368%bnl2/bias(parameters)_139765161132456*
shape[]
X
Threshold_139765161298344ThresholdBackwardBatchNorm_139765161085752*
shape[]
T
Dropout_139765161298584DropoutBackwardThreshold_139765161298344*
shape[]
L
%l3/weight(parameters)_139765161398208AccumulateGrad*
shape
[2, 128]
d
Transpose_139765161298824TransposeBackward%l3/weight(parameters)_139765161398208*
shape[]
�
Addmm_139765161299064AddmmBackward#l3/bias(parameters)_139765161132104Dropout_139765161298584Transpose_139765161298824*
shape[]"�f��#       ��wC	��f�t�A*

training_lossDAwg�t'       ��F	}��f�t�A*

training_accuracy%��>���