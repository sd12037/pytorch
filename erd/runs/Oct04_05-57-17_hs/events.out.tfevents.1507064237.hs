	       7�q9	+(Xk�t�A�r��#      ;�	�1l�t�A"�$
E
#l3/bias(parameters)_140452102271048AccumulateGrad*
shape[2]
G
#l2/bias(parameters)_140452102537064AccumulateGrad*
shape[128]
G
#l1/bias(parameters)_140452102536888AccumulateGrad*
shape[512]
<
inputTensor_140452086182744AccumulateGrad*
shape[]
T
(conv1/weight(parameters)_140450464193968AccumulateGrad*
shape[64, 1, 2, 2]
�
ConvNd_140452086182576ConvNdBackwardinputTensor_140452086182744(conv1/weight(parameters)_140450464193968NoneType140452242975952*
shape[]
_
FeatureDropout_140450489240216FeatureDropoutBackwardConvNd_140452086182576*
shape[]
]
MaxPool2d_140450489240456MaxPool2dBackwardFeatureDropout_140450489240216*
shape[]
V
(conv2/weight(parameters)_140452102488136AccumulateGrad*
shape[128, 64, 2, 2]
�
ConvNd_140452086182184ConvNdBackwardMaxPool2d_140450489240456(conv2/weight(parameters)_140452102488136NoneType140452242975952*
shape[]
_
FeatureDropout_140450489240696FeatureDropoutBackwardConvNd_140452086182184*
shape[]
]
MaxPool2d_140450489240936MaxPool2dBackwardFeatureDropout_140450489240696*
shape[]
U
(conv3/weight(parameters)_140452102488224AccumulateGrad*
shape[1, 128, 2, 2]
�
ConvNd_140452102225872ConvNdBackwardMaxPool2d_140450489240936(conv3/weight(parameters)_140452102488224NoneType140452242975952*
shape[]
_
FeatureDropout_140450489241176FeatureDropoutBackwardConvNd_140452102225872*
shape[]
m
!AdaptiveMaxPool2d_140452102434888AdaptiveMaxPool2dBackwardFeatureDropout_140450489241176*
shape[]
\
Squeeze_140452102435128SqueezeBackward!AdaptiveMaxPool2d_140452102434888*
shape[]
S
-lstm/weight_ih_l0(parameters)_140452102535920AccumulateGrad*
shape	[40, 5]
T
-lstm/weight_hh_l0(parameters)_140452102536008AccumulateGrad*
shape
[40, 10]
N
+lstm/bias_ih_l0(parameters)_140452102536096AccumulateGrad*
shape[40]
N
+lstm/bias_hh_l0(parameters)_140452102536360AccumulateGrad*
shape[40]
[
5lstm/weight_ih_l0_reverse(parameters)_140452102536448AccumulateGrad*
shape	[40, 5]
\
5lstm/weight_hh_l0_reverse(parameters)_140452102536536AccumulateGrad*
shape
[40, 10]
V
3lstm/bias_ih_l0_reverse(parameters)_140452102536624AccumulateGrad*
shape[40]
V
3lstm/bias_hh_l0_reverse(parameters)_140452102536712AccumulateGrad*
shape[40]
�
CudnnRNN140452102435368CudnnRNNSqueeze_140452102435128-lstm/weight_ih_l0(parameters)_140452102535920-lstm/weight_hh_l0(parameters)_140452102536008+lstm/bias_ih_l0(parameters)_140452102536096+lstm/bias_hh_l0(parameters)_1404521025363605lstm/weight_ih_l0_reverse(parameters)_1404521025364485lstm/weight_hh_l0_reverse(parameters)_1404521025365363lstm/bias_ih_l0_reverse(parameters)_1404521025366243lstm/bias_hh_l0_reverse(parameters)_140452102536712NoneType140452242975952NoneType140452242975952*
shape[]
L
View_140452102435608ViewBackwardCudnnRNN140452102435368*
shape[]
N
%l1/weight(parameters)_140452102536800AccumulateGrad*
shape
[512, 100]
d
Transpose_140452102435848TransposeBackward%l1/weight(parameters)_140452102536800*
shape[]
�
Addmm_140452102436088AddmmBackward#l1/bias(parameters)_140452102536888View_140452102435608Transpose_140452102435848*
shape[]
T
Threshold_140452102436328ThresholdBackwardAddmm_140452102436088*
shape[]
T
Dropout_140452102436568DropoutBackwardThreshold_140452102436328*
shape[]
N
%l2/weight(parameters)_140452102536976AccumulateGrad*
shape
[128, 512]
d
Transpose_140452102436808TransposeBackward%l2/weight(parameters)_140452102536976*
shape[]
�
Addmm_140452102437048AddmmBackward#l2/bias(parameters)_140452102537064Dropout_140452102436568Transpose_140452102436808*
shape[]
K
'bnl2/weight(parameters)_140452102271312AccumulateGrad*
shape[128]
I
%bnl2/bias(parameters)_140452102271400AccumulateGrad*
shape[128]
�
BatchNorm_140452102224696BatchNormBackwardAddmm_140452102437048'bnl2/weight(parameters)_140452102271312%bnl2/bias(parameters)_140452102271400*
shape[]
X
Threshold_140452102437288ThresholdBackwardBatchNorm_140452102224696*
shape[]
T
Dropout_140452102437528DropoutBackwardThreshold_140452102437288*
shape[]
L
%l3/weight(parameters)_140452102537152AccumulateGrad*
shape
[2, 128]
d
Transpose_140452102437768TransposeBackward%l3/weight(parameters)_140452102537152*
shape[]
�
Addmm_140452102438008AddmmBackward#l3/bias(parameters)_140452102271048Dropout_140452102437528Transpose_140452102437768*
shape[]"��;#       ��wC	jhl�t�A*

training_loss�AA��'       ��F	pkhl�t�A*

training_accuracyn�?����#       ��wC	&ʭl�t�A*

training_lossqyAhi�'       ��F	�˭l�t�A*

training_accuracyI2?��w�#       ��wC	���l�t�A*

training_loss�tA���'       ��F	���l�t�A*

training_accuracy��?�k�#       ��wC	u�<m�t�A*

training_loss�eA�k�n'       ��F	��<m�t�A*

training_accuracy%	?Iz��#       ��wC	R�m�t�A*

training_loss%)AK�/�'       ��F	���m�t�A*

training_accuracyI�!?�s��%       �6�	�͚m�t�A*

validation_lossoj�@�7�2)       7�_ 	hϚm�t�A*

validation_accuracyn[W?��:#       ��wC	���m�t�A*

training_loss�A��,'       ��F	���m�t�A*

training_accuracyn�$?��,#       ��wC	��)n�t�A*

training_loss�Ao�V'       ��F	"�)n�t�A*

training_accuracy  *?��>h#       ��wC	V�on�t�A*

training_loss�wAP׺�'       ��F	��on�t�A*

training_accuracyIR,?#	�#       ��wC	�p�n�t�A	*

training_loss1�A2�9�'       ��F	r�n�t�A	*

training_accuracyn;/?����#       ��wC	�'�n�t�A
*

training_loss���@|�r'       ��F	)�n�t�A
*

training_accuracy%I0?4��<%       �6�	�o�t�A
*

validation_loss���@�w�8)       7�_ 	*o�t�A
*

validation_accuracyn;U?G\#       ��wC	��^o�t�A*

training_loss�+�@���'       ��F	��^o�t�A*

training_accuracy��1?�yi#       ��wC	o��o�t�A*

training_loss;��@��3)'       ��F	ߺ�o�t�A*

training_accuracy `2?k�s8#       ��wC	~(�o�t�A*

training_loss�H�@L��G'       ��F	�)�o�t�A*

training_accuracy�d3?�4;#       ��wC	��3p�t�A*

training_loss\��@�d�&'       ��F	-�3p�t�A*

training_accuracyn;3?�/��#       ��wC	��yp�t�A*

training_loss��@es0�'       ��F	}�yp�t�A*

training_accuracyn[2?�O�e%       �6�	�ڑp�t�A*

validation_loss�<�@�N��)       7�_ 	ܑp�t�A*

validation_accuracy��U?	�##       ��wC	�c�p�t�A*

training_loss+��@���'       ��F	�d�p�t�A*

training_accuracy�m3?D�L#       ��wC	�;"q�t�A*

training_loss�2�@�Av�'       ��F	="q�t�A*

training_accuracyۖ2?s��#       ��wC	��hq�t�A*

training_loss��@�>�'       ��F	x�hq�t�A*

training_accuracy�d3?4�"#       ��wC	!�q�t�A*

training_loss,]�@�@�'       ��F	���q�t�A*

training_accuracy۶5?l�pi#       ��wC	bE�q�t�A*

training_lossl��@�uܱ'       ��F	�F�q�t�A*

training_accuracy�V4?x�O5%       �6�	�?r�t�A*

validation_loss�j�@�ϓ)       7�_ 	�@r�t�A*

validation_accuracy �V?q��#       ��wC	��Vr�t�A*

training_loss�!�@�=��'       ��F	O�Vr�t�A*

training_accuracyn�4?`��#       ��wC	���r�t�A*

training_loss��@�]�;'       ��F	h��r�t�A*

training_accuracy%�3?&�D�#       ��wC	_��r�t�A*

training_loss�y�@2wV�'       ��F	Ƙ�r�t�A*

training_accuracyn�4?<�z#       ��wC	z�-s�t�A*

training_lossLO�@��u'       ��F	��-s�t�A*

training_accuracy�$5?���=#       ��wC	��us�t�A*

training_loss���@?5�'       ��F	��us�t�A*

training_accuracy @4?���R%       �6�	�?�s�t�A*

validation_loss_�@WnP)       7�_ 	HA�s�t�A*

validation_accuracy �V?�9�#       ��wC	t�s�t�A*

training_loss���@�n�'       ��F	�u�s�t�A*

training_accuracy �5?�&�=#       ��wC	� t�t�A*

training_lossXh�@"<��'       ��F	A t�t�A*

training_accuracy�v5?v�<#       ��wC	��gt�t�A*

training_loss�P�@e���'       ��F	W�gt�t�A*

training_accuracy�d6?��(#       ��wC	ت�t�t�A*

training_loss��@��k'       ��F	i��t�t�A*

training_accuracy�v6?6m��#       ��wC	��t�t�A*

training_loss��@�Hl'       ��F	?�t�t�A*

training_accuracy�5?�/[�%       �6�	�u�t�A*

validation_loss��@���)       7�_ 	[�u�t�A*

validation_accuracyn�V?`��
#       ��wC	�Zu�t�A*

training_lossqW�@L��'       ��F	L�Zu�t�A*

training_accuracy%)5?��#       ��wC	n��u�t�A *

training_losse�@Z��'       ��F	ڽ�u�t�A *

training_accuracy��6?F�~#       ��wC	4��u�t�A!*

training_loss��@�l�'       ��F	���u�t�A!*

training_accuracy��6?�w�#       ��wC	4r0v�t�A"*

training_loss���@%ɱ�'       ��F	�s0v�t�A"*

training_accuracyI�5?`%�#       ��wC	J3xv�t�A#*

training_lossG��@���$'       ��F	�4xv�t�A#*

training_accuracyI�7?[1��%       �6�	K�v�t�A#*

validation_loss�߱@��\l)       7�_ 	��v�t�A#*

validation_accuracy۶W?��#       ��wC	P��v�t�A$*

training_loss3��@��v'       ��F	���v�t�A$*

training_accuracy%)6?)�ն#       ��wC	W�w�t�A%*

training_loss��@�$.
'       ��F	��w�t�A%*

training_accuracy��6?�}N#       ��wC	�9gw�t�A&*

training_lossD�@��N�'       ��F	*;gw�t�A&*

training_accuracyn[7?��#       ��wC	��w�t�A'*

training_loss�k�@N1W�'       ��F	{��w�t�A'*

training_accuracy%�6?�@!#       ��wC	���w�t�A(*

training_loss�,�@�)�'       ��F	ʢ�w�t�A(*

training_accuracy @6?g4��%       �6�	��x�t�A(*

validation_loss9��@�|��)       7�_ 	"�x�t�A(*

validation_accuracy%	X?�:��#       ��wC	�NTx�t�A)*

training_loss���@�U�5'       ��F	IPTx�t�A)*

training_accuracyn�6?�
�0#       ��wC	כx�t�A**

training_loss��@��k�'       ��F	y؛x�t�A**

training_accuracy%I7?ԝ�#       ��wC	�"�x�t�A+*

training_lossv�@)j�'       ��F	\$�x�t�A+*

training_accuracyn�8?rU##       ��wC	5�,y�t�A,*

training_loss+��@v\Њ'       ��F	z�,y�t�A,*

training_accuracy��7?���#       ��wC	d�ry�t�A-*

training_lossu�@��BB'       ��F	��ry�t�A-*

training_accuracyn[:?ۥq%       �6�	��y�t�A-*

validation_loss�@�.�)       7�_ 	�y�t�A-*

validation_accuracy��V?zi�#       ��wC	�v�y�t�A.*

training_loss �@�'       ��F	x�y�t�A.*

training_accuracy�d9?�+Ld#       ��wC	��z�t�A/*

training_loss���@ɴ��'       ��F	/�z�t�A/*

training_accuracy%I9?Pl�>#       ��wC	��bz�t�A0*

training_loss$O�@�`]^'       ��F	x�bz�t�A0*

training_accuracy�69?��)�#       ��wC	>ɨz�t�A1*

training_loss ��@W�Y�'       ��F	ʨz�t�A1*

training_accuracy%�8?����#       ��wC	�`�z�t�A2*

training_loss�c�@���'       ��F	�a�z�t�A2*

training_accuracyIR9?����%       �6�	ȑ	{�t�A2*

validation_loss �@�?��)       7�_ 	��	{�t�A2*

validation_accuracyn;Q?��i}#       ��wC	1P{�t�A3*

training_lossu�@�_�'       ��F	�P{�t�A3*

training_accuracyI�9?w(�S#       ��wC	�f�{�t�A4*

training_loss��@�d�#'       ��F	Nh�{�t�A4*

training_accuracy%�8?0�#       ��wC	�?�{�t�A5*

training_loss8��@۔H'       ��F	kA�{�t�A5*

training_accuracyn�9?Ƣ��#       ��wC	�>)|�t�A6*

training_loss�x�@[�V�'       ��F	"@)|�t�A6*

training_accuracy @9?�1�#       ��wC	Y{o|�t�A7*

training_loss(��@{��B'       ��F	~|o|�t�A7*

training_accuracy۶:?m�ۊ%       �6�	Yq�|�t�A7*

validation_loss�ͱ@ZF�)       7�_ 	�r�|�t�A7*

validation_accuracyn[M?|�Y�#       ��wC	y]�|�t�A8*

training_loss���@9�'       ��F	�^�|�t�A8*

training_accuracy��:?����#       ��wC	��}�t�A9*

training_loss�+�@o_�'       ��F	!�}�t�A9*

training_accuracyI�;?��2�#       ��wC	:_}�t�A:*

training_lossѵ�@ � �'       ��F	l;_}�t�A:*

training_accuracy%):?�XB�#       ��wC	
��}�t�A;*

training_loss���@�8b'       ��F	q��}�t�A;*

training_accuracy�m9?�Z�#       ��wC	���}�t�A<*

training_loss#��@>�q�'       ��F	;��}�t�A<*

training_accuracy @9?�)%       �6�	ĥ~�t�A<*

validation_lossHl�@&��)       7�_ 	*�~�t�A<*

validation_accuracy�VJ?���/#       ��wC	�L~�t�A=*

training_loss���@~KW�'       ��F	��L~�t�A=*

training_accuracy��:?�R�#       ��wC	�ҕ~�t�A>*

training_loss@��@����'       ��F	aԕ~�t�A>*

training_accuracyn�<?�]�#       ��wC	/�~�t�A?*

training_lossU(�@��0�'       ��F	l0�~�t�A?*

training_accuracyn�;?�'r#       ��wC	^�#�t�A@*

training_loss�O�@�'       ��F	Ć#�t�A@*

training_accuracy��9?���j#       ��wC	��l�t�AA*

training_loss���@#;��'       ��F	�l�t�AA*

training_accuracy �;?S<`%       �6�	�Ȅ�t�AA*

validation_loss�q�@�K%)       7�_ 	<ʄ�t�AA*

validation_accuracy �I?`���#       ��wC	����t�AB*

training_lossΎ�@wb'       ��F	e���t�AB*

training_accuracy%);?F�p#       ��wC	���t�AC*

training_lossy%�@����'       ��F	b��t�AC*

training_accuracy  ;?/�tB#       ��wC	�[��t�AD*

training_loss��@��$'       ��F	q�[��t�AD*

training_accuracyI�:?B##       ��wC	5���t�AE*

training_loss� �@�CԵ'       ��F	u6���t�AE*

training_accuracy  :?�H�6#       ��wC	���t�AF*

training_loss���@N|��'       ��F	���t�AF*

training_accuracyI�;?E���%       �6�	A���t�AF*

validation_lossD��@��\)       7�_ 	����t�AF*

validation_accuracy �H?8B�/#       ��wC	�J��t�AG*

training_loss(��@�Ȩ�'       ��F	JJ��t�AG*

training_accuracy�m;?En�#       ��wC	W����t�AH*

training_loss��@��d*'       ��F	�����t�AH*

training_accuracy�<?8g�j#       ��wC	Զځ�t�AI*

training_lossW��@ɫ8y'       ��F	9�ځ�t�AI*

training_accuracy%�;?<\��#       ��wC	�z!��t�AJ*

training_loss��@t]'       ��F	K|!��t�AJ*

training_accuracy�;?�Yo�#       ��wC	=�j��t�AK*

training_loss�l�@8Ʃ�'       ��F	��j��t�AK*

training_accuracy�V;?q�E�%       �6�	🂂�t�AK*

validation_lossTа@@�l	)       7�_ 	]����t�AK*

validation_accuracy%IH?��W#       ��wC	IFɂ�t�AL*

training_lossH�@�핃'       ��F	�Gɂ�t�AL*

training_accuracyI�;?H��#       ��wC	����t�AM*

training_lossh�@�HϘ'       ��F	���t�AM*

training_accuracy��<?'�^�#       ��wC	;�V��t�AN*

training_lossD�@&��'       ��F	��V��t�AN*

training_accuracy%�;?B$#       ��wC	Dៃ�t�AO*

training_losso��@�c�M'       ��F	�⟃�t�AO*

training_accuracy�$=?wuDy#       ��wC	���t�AP*

training_loss11�@��a'       ��F	Q���t�AP*

training_accuracyI�<?�G%       �6�	�� ��t�AP*

validation_lossw��@���)       7�_ 	� ��t�AP*

validation_accuracy��I?�>x#       ��wC	�H��t�AQ*

training_loss�O�@�j�'       ��F	H��t�AQ*

training_accuracy%�<?�ހ�#       ��wC	䅏��t�AR*

training_loss��@����'       ��F	C����t�AR*

training_accuracy��<?m���#       ��wC	\Z؄�t�AS*

training_loss�
�@����'       ��F	�[؄�t�AS*

training_accuracy �<?�q�#       ��wC	���t�AT*

training_loss�c�@1�'       ��F	}���t�AT*

training_accuracy%I=?j'�#       ��wC	��f��t�AU*

training_lossI��@ n��'       ��F	C�f��t�AU*

training_accuracyn{=?P|��%       �6�	�3��t�AU*

validation_lossp8�@JA I)       7�_ 	�4��t�AU*

validation_accuracyn�I?	~�#       ��wC	�Ņ�t�AV*

training_loss ��@�^�'       ��F	r�Ņ�t�AV*

training_accuracy��=?44h�#       ��wC	�"��t�AW*

training_loss���@r�'       ��F	X$��t�AW*

training_accuracy��=?�y!#       ��wC	�T��t�AX*

training_loss��@"~H'       ��F	N�T��t�AX*

training_accuracyn;=?�zZ�#       ��wC	�a���t�AY*

training_loss��@�.:�'       ��F	c���t�AY*

training_accuracyn�=?���
#       ��wC	j��t�AZ*

training_loss��@���'       ��F	���t�AZ*

training_accuracyI�=?�,3%       �6�	�=���t�AZ*

validation_loss���@̕��)       7�_ 	5?���t�AZ*

validation_accuracy @J?4�L#       ��wC	�rF��t�A[*

training_loss��@���r'       ��F	 tF��t�A[*

training_accuracy%�<?�C�G#       ��wC	�����t�A\*

training_loss���@d]y�'       ��F	惎��t�A\*

training_accuracy�=?7v'#       ��wC	�؇�t�A]*

training_lossp$�@���X'       ��F	�؇�t�A]*

training_accuracy�d>?(SHQ#       ��wC	T� ��t�A^*

training_lossGL�@��.'       ��F	�� ��t�A^*

training_accuracy  <?ybr#       ��wC	gpj��t�A_*

training_lossJ��@l���'       ��F	�qj��t�A_*

training_accuracyI2@?g�%       �6�	�����t�A_*

validation_loss(�@}��)       7�_ 	����t�A_*

validation_accuracyn{J?Z#       ��wC	| ʈ�t�A`*

training_lossz��@T}~�'       ��F	�!ʈ�t�A`*

training_accuracyI�>?K�0�#       ��wC	�K��t�Aa*

training_loss��@�A�'       ��F	M��t�Aa*

training_accuracy��>?��l�#       ��wC	elX��t�Ab*

training_loss,r�@Hstn'       ��F	�mX��t�Ab*

training_accuracy `>?�͞=#       ��wC	H����t�Ac*

training_lossƓ�@����'       ��F	�����t�Ac*

training_accuracy �>?��i#       ��wC	����t�Ad*

training_loss���@YA�a'       ��F	���t�Ad*

training_accuracy�d=?.�5�%       �6�	�� ��t�Ad*

validation_loss$��@ܐtS)       7�_ 	L� ��t�Ad*

validation_accuracy�K?^zE@