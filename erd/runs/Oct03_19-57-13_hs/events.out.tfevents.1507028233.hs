	       7�q9	��`B�t�A�&#      ;�	��#C�t�A"�$
E
#l3/bias(parameters)_140019829801120AccumulateGrad*
shape[2]
G
#l2/bias(parameters)_140019830071232AccumulateGrad*
shape[128]
G
#l1/bias(parameters)_140019830071056AccumulateGrad*
shape[512]
<
inputTensor_140019829498992AccumulateGrad*
shape[]
T
(conv1/weight(parameters)_140018231631368AccumulateGrad*
shape[64, 1, 2, 2]
�
ConvNd_140019829498824ConvNdBackwardinputTensor_140019829498992(conv1/weight(parameters)_140018231631368NoneType140020010532048*
shape[]
_
FeatureDropout_140018256681624FeatureDropoutBackwardConvNd_140019829498824*
shape[]
]
MaxPool2d_140018256681864MaxPool2dBackwardFeatureDropout_140018256681624*
shape[]
V
(conv2/weight(parameters)_140019830018208AccumulateGrad*
shape[128, 64, 2, 2]
�
ConvNd_140019829498432ConvNdBackwardMaxPool2d_140018256681864(conv2/weight(parameters)_140019830018208NoneType140020010532048*
shape[]
_
FeatureDropout_140018256682104FeatureDropoutBackwardConvNd_140019829498432*
shape[]
]
MaxPool2d_140018256682344MaxPool2dBackwardFeatureDropout_140018256682104*
shape[]
U
(conv3/weight(parameters)_140019830018296AccumulateGrad*
shape[1, 128, 2, 2]
�
ConvNd_140019829498152ConvNdBackwardMaxPool2d_140018256682344(conv3/weight(parameters)_140019830018296NoneType140020010532048*
shape[]
_
FeatureDropout_140018256682584FeatureDropoutBackwardConvNd_140019829498152*
shape[]
m
!AdaptiveMaxPool2d_140019829968968AdaptiveMaxPool2dBackwardFeatureDropout_140018256682584*
shape[]
\
Squeeze_140019829969208SqueezeBackward!AdaptiveMaxPool2d_140019829968968*
shape[]
S
-lstm/weight_ih_l0(parameters)_140019830070088AccumulateGrad*
shape	[40, 5]
T
-lstm/weight_hh_l0(parameters)_140019830070176AccumulateGrad*
shape
[40, 10]
N
+lstm/bias_ih_l0(parameters)_140019830070264AccumulateGrad*
shape[40]
N
+lstm/bias_hh_l0(parameters)_140019830070528AccumulateGrad*
shape[40]
[
5lstm/weight_ih_l0_reverse(parameters)_140019830070616AccumulateGrad*
shape	[40, 5]
\
5lstm/weight_hh_l0_reverse(parameters)_140019830070704AccumulateGrad*
shape
[40, 10]
V
3lstm/bias_ih_l0_reverse(parameters)_140019830070792AccumulateGrad*
shape[40]
V
3lstm/bias_hh_l0_reverse(parameters)_140019830070880AccumulateGrad*
shape[40]
�
CudnnRNN140019829969448CudnnRNNSqueeze_140019829969208-lstm/weight_ih_l0(parameters)_140019830070088-lstm/weight_hh_l0(parameters)_140019830070176+lstm/bias_ih_l0(parameters)_140019830070264+lstm/bias_hh_l0(parameters)_1400198300705285lstm/weight_ih_l0_reverse(parameters)_1400198300706165lstm/weight_hh_l0_reverse(parameters)_1400198300707043lstm/bias_ih_l0_reverse(parameters)_1400198300707923lstm/bias_hh_l0_reverse(parameters)_140019830070880NoneType140020010532048NoneType140020010532048*
shape[]
L
View_140019829969688ViewBackwardCudnnRNN140019829969448*
shape[]
N
%l1/weight(parameters)_140019830070968AccumulateGrad*
shape
[512, 100]
d
Transpose_140019829969928TransposeBackward%l1/weight(parameters)_140019830070968*
shape[]
�
Addmm_140019829970168AddmmBackward#l1/bias(parameters)_140019830071056View_140019829969688Transpose_140019829969928*
shape[]
T
Threshold_140019829970408ThresholdBackwardAddmm_140019829970168*
shape[]
T
Dropout_140019829970648DropoutBackwardThreshold_140019829970408*
shape[]
N
%l2/weight(parameters)_140019830071144AccumulateGrad*
shape
[128, 512]
d
Transpose_140019829970888TransposeBackward%l2/weight(parameters)_140019830071144*
shape[]
�
Addmm_140019829971128AddmmBackward#l2/bias(parameters)_140019830071232Dropout_140019829970648Transpose_140019829970888*
shape[]
K
'bnl2/weight(parameters)_140019829801384AccumulateGrad*
shape[128]
I
%bnl2/bias(parameters)_140019829801472AccumulateGrad*
shape[128]
�
BatchNorm_140019830275152BatchNormBackwardAddmm_140019829971128'bnl2/weight(parameters)_140019829801384%bnl2/bias(parameters)_140019829801472*
shape[]
X
Threshold_140019829971368ThresholdBackwardBatchNorm_140019830275152*
shape[]
T
Dropout_140019829971608DropoutBackwardThreshold_140019829971368*
shape[]
L
%l3/weight(parameters)_140019829801032AccumulateGrad*
shape
[2, 128]
d
Transpose_140019829971848TransposeBackward%l3/weight(parameters)_140019829801032*
shape[]
�
Addmm_140019829972088AddmmBackward#l3/bias(parameters)_140019829801120Dropout_140019829971608Transpose_140019829971848*
shape[]"w���#       ��wC	�RlC�t�A*

training_loss�4A��n'       ��F	�TlC�t�A*

training_accuracyI�?�>�Y#       ��wC	��C�t�A*

training_lossp�A1gO�'       ��F	b�C�t�A*

training_accuracyn?�:a�#       ��wC	���C�t�A*

training_loss$�A0�i&'       ��F	���C�t�A*

training_accuracy�-#?�6
�#       ��wC	p�AD�t�A*

training_loss��
A�6T�'       ��F	�AD�t�A*

training_accuracy  (?�	�b#       ��wC	���D�t�A*

training_loss�AQpY�'       ��F	g��D�t�A*

training_accuracy�$,?�ܱ�%       �6�	"ޞD�t�A*

validation_lossY��@F���)       7�_ 	�ߞD�t�A*

validation_accuracy%�R?8�i6#       ��wC	��D�t�A*

training_lossk�A77&�'       ��F	��D�t�A*

training_accuracy  .?B>[�#       ��wC	0�-E�t�A*

training_loss��A���J'       ��F	��-E�t�A*

training_accuracy%I0?�NJm#       ��wC	!1tE�t�A*

training_loss�A��y'       ��F	�2tE�t�A*

training_accuracy��/?4�zp#       ��wC	��E�t�A	*

training_lossk��@lo%'       ��F	b �E�t�A	*

training_accuracy�1?���#       ��wC	��F�t�A
*

training_loss<z�@,i4b'       ��F	3�F�t�A
*

training_accuracy��1?����%       �6�	XuF�t�A
*

validation_lossHL�@He��)       7�_ 	�vF�t�A
*

validation_accuracy��S?�ͼ#       ��wC	��cF�t�A*

training_loss���@��z'       ��F	1�cF�t�A*

training_accuracyn[1?���#       ��wC	�F�t�A*

training_loss�c�@��s�'       ��F	��F�t�A*

training_accuracyۖ2?p���#       ��wC	�l�F�t�A*

training_loss&��@k��'       ��F	`n�F�t�A*

training_accuracy�62?��w�#       ��wC	{�8G�t�A*

training_lossJ��@j�'       ��F	��8G�t�A*

training_accuracyn{2?%S�Z#       ��wC	Oy�G�t�A*

training_loss��@Z�r'       ��F	�z�G�t�A*

training_accuracy%�2?���%       �6�	ƒ�G�t�A*

validation_loss�{�@��)       7�_ 	3��G�t�A*

validation_accuracy۶T?����#       ��wC	D��G�t�A*

training_loss�P�@��]r'       ��F	���G�t�A*

training_accuracy��1?��=#       ��wC	i(H�t�A*

training_loss3�@@� 2'       ��F	�j(H�t�A*

training_accuracyIR2?<a�#       ��wC	�0nH�t�A*

training_loss'��@Ҷ<�'       ��F	2nH�t�A*

training_accuracy��0?��B~#       ��wC	��H�t�A*

training_loss���@!nR�'       ��F	��H�t�A*

training_accuracyn�2?���#       ��wC	�H�t�A*

training_loss���@cv�c'       ��F	m�H�t�A*

training_accuracy۶2?����%       �6�	�BI�t�A*

validation_loss�K�@7��)       7�_ 	/DI�t�A*

validation_accuracy�VU?��+#       ��wC	��[I�t�A*

training_loss��@9���'       ��F	"�[I�t�A*

training_accuracyI�3?�(�#       ��wC	�ܠI�t�A*

training_loss�o�@�R�,'       ��F	�ޠI�t�A*

training_accuracy �3?�[ѓ#       ��wC	h��I�t�A*

training_lossb��@Do��'       ��F	���I�t�A*

training_accuracy �3?�9Y%#       ��wC	��/J�t�A*

training_lossX�@�'��'       ��F	 �/J�t�A*

training_accuracyn;3?<�/�#       ��wC	wAzJ�t�A*

training_loss���@�0'       ��F	�BzJ�t�A*

training_accuracyI�3?��%P%       �6�	��J�t�A*

validation_loss��@F�I)       7�_ 	c�J�t�A*

validation_accuracyI�U?��Q�#       ��wC	�G�J�t�A*

training_lossֈ�@��,='       ��F	�H�J�t�A*

training_accuracyn;4?���:#       ��wC	> $K�t�A*

training_lossL��@3tV�'       ��F	�$K�t�A*

training_accuracyI�3?ʹ�#       ��wC	m�lK�t�A*

training_loss< �@���'       ��F	��lK�t�A*

training_accuracyۖ3?�b�e#       ��wC	tn�K�t�A*

training_loss��@���'       ��F	�o�K�t�A*

training_accuracy �3?�J��#       ��wC	"VL�t�A*

training_lossй�@�@�/'       ��F	fWL�t�A*

training_accuracy �4? y�%       �6�	V�L�t�A*

validation_loss���@��o�)       7�_ 	�L�t�A*

validation_accuracy  V?fA&�#       ��wC	�4cL�t�A*

training_lossYr�@/X7�'       ��F	x6cL�t�A*

training_accuracy�d4?A�
�#       ��wC	Q0�L�t�A *

training_loss��@Jñ'       ��F	�1�L�t�A *

training_accuracy۶5?�ǖC#       ��wC	�C�L�t�A!*

training_loss���@G=E�'       ��F	2E�L�t�A!*

training_accuracy��5??�;O#       ��wC	J:=M�t�A"*

training_lossh(�@��u9'       ��F	�;=M�t�A"*

training_accuracy��4? �|�#       ��wC	P�M�t�A#*

training_loss���@�"6\'       ��F	��M�t�A#*

training_accuracy%�4?��[j%       �6�	k�M�t�A#*

validation_loss_��@��[�)       7�_ 	~l�M�t�A#*

validation_accuracy%�V?����#       ��wC	�i�M�t�A$*

training_loss���@w�'       ��F	�k�M�t�A$*

training_accuracy�D5?�<�#       ��wC	�])N�t�A%*

training_loss1C�@L��J'       ��F	_)N�t�A%*

training_accuracy%�5?��>#       ��wC	�3oN�t�A&*

training_lossz2�@����'       ��F	5oN�t�A&*

training_accuracy��5?���#       ��wC	�s�N�t�A'*

training_lossE��@rm'-'       ��F	Vu�N�t�A'*

training_accuracy��6?�?ޝ#       ��wC	��N�t�A(*

training_losshI�@��'       ��F	X �N�t�A(*

training_accuracy�-5?��%%       �6�	+KO�t�A(*

validation_loss�^�@,�v )       7�_ 	�LO�t�A(*

validation_accuracyn[W?��"#       ��wC	�ZO�t�A)*

training_loss��@+w�'       ��F	hZO�t�A)*

training_accuracy �6?'S#       ��wC	���O�t�A**

training_loss.��@����'       ��F	���O�t�A**

training_accuracy��6?�I3�#       ��wC	�W�O�t�A+*

training_lossg^�@.Y+�'       ��F	�X�O�t�A+*

training_accuracy�v6?M�gO#       ��wC	�n1P�t�A,*

training_loss��@�R��'       ��F	|p1P�t�A,*

training_accuracy�7?ڈ�#       ��wC	:�wP�t�A-*

training_loss3Z�@##,'       ��F	��wP�t�A-*

training_accuracy��6?�Ǹ%       �6�	A*�P�t�A-*

validation_loss�q�@hf�T)       7�_ 	�+�P�t�A-*

validation_accuracyI�V?��#       ��wC	�D�P�t�A.*

training_loss��@��'       ��F	pF�P�t�A.*

training_accuracy�7?��G#       ��wC	,�Q�t�A/*

training_lossh��@���'       ��F	~�Q�t�A/*

training_accuracy%)8?vTg#       ��wC		�hQ�t�A0*

training_loss��@vn�'       ��F	R�hQ�t�A0*

training_accuracyn�7?�%�m#       ��wC	��Q�t�A1*

training_lossp�@��`"'       ��F	a�Q�t�A1*

training_accuracyI�7?���#       ��wC	�J�Q�t�A2*

training_loss�M�@�~�C'       ��F	<L�Q�t�A2*

training_accuracy%�8?�XY�%       �6�	
�R�t�A2*

validation_loss޴@�L)       7�_ 	t�R�t�A2*

validation_accuracy `R?a7�#       ��wC	�{VR�t�A3*

training_loss�`�@}�-�'       ��F	^}VR�t�A3*

training_accuracy%i8?��^�#       ��wC	S��R�t�A4*

training_loss�6�@�w��'       ��F	���R�t�A4*

training_accuracy��8?�чv#       ��wC	f[�R�t�A5*

training_lossȠ�@��O�'       ��F	�\�R�t�A5*

training_accuracy�M8?�)� #       ��wC	�/S�t�A6*

training_lossc��@��J'       ��F	/S�t�A6*

training_accuracy%�:?U�66#       ��wC	K�uS�t�A7*

training_loss���@�ADe'       ��F	��uS�t�A7*

training_accuracyI�8?)%%       �6�	b��S�t�A7*

validation_loss�Ӵ@୅)       7�_ 	� �S�t�A7*

validation_accuracyn�O?=B'�#       ��wC	bk�S�t�A8*

training_loss���@d�/�'       ��F	�l�S�t�A8*

training_accuracy �:?Y�a�#       ��wC	�XT�t�A9*

training_lossP!�@6ˤ�'       ��F	SZT�t�A9*

training_accuracyIR:?��#       ��wC	ΌcT�t�A:*

training_loss$��@ �[�'       ��F	:�cT�t�A:*

training_accuracy%I:?M��#       ��wC	���T�t�A;*

training_lossf��@i'�'       ��F	'��T�t�A;*

training_accuracyI�9?O��#       ��wC	���T�t�A<*

training_loss���@K���'       ��F	���T�t�A<*

training_accuracy�V<?��^%       �6�	��
U�t�A<*

validation_loss�#�@����)       7�_ 	N�
U�t�A<*

validation_accuracy �L?���#       ��wC	xRU�t�A=*

training_lossSa�@)N��'       ��F	�RU�t�A=*

training_accuracyIR:?��z�#       ��wC	?��U�t�A>*

training_lossѼ�@>+�'       ��F	���U�t�A>*

training_accuracyn�:?�N!�#       ��wC	�)�U�t�A?*

training_loss���@�0�'       ��F	+�U�t�A?*

training_accuracy�<? ; R#       ��wC	:(V�t�A@*

training_loss}�@J�Z�'       ��F	�;(V�t�A@*

training_accuracy۶;?�O�A#       ��wC	f,qV�t�AA*

training_lossd�@{~�s'       ��F	�-qV�t�AA*

training_accuracyI�<?���%       �6�	�f�V�t�AA*

validation_loss�ζ@�뻼)       7�_ 	`h�V�t�AA*

validation_accuracy  J?���#       ��wC	�C�V�t�AB*

training_loss~��@%��P'       ��F	�D�V�t�AB*

training_accuracy�d;?h��#       ��wC	��W�t�AC*

training_lossM��@�.Fd'       ��F	�W�t�AC*

training_accuracy��;? V�#       ��wC	�B^W�t�AD*

training_loss�"�@R_V'       ��F	FD^W�t�AD*

training_accuracy �<?�糲#       ��wC	޴�W�t�AE*

training_loss�Y�@XO�'       ��F	F��W�t�AE*

training_accuracy��<?`���#       ��wC	^��W�t�AF*

training_loss�:�@��'       ��F	���W�t�AF*

training_accuracyI�;?>	b�%       �6�	bxX�t�AF*

validation_loss�4�@�!��)       7�_ 	�yX�t�AF*

validation_accuracyI�I?Z�V#       ��wC	�$NX�t�AG*

training_loss���@���'       ��F	=&NX�t�AG*

training_accuracy��<?�6�z#       ��wC	��X�t�AH*

training_loss1�@��T'       ��F	'�X�t�AH*

training_accuracy��<?���#       ��wC	�D�X�t�AI*

training_loss|��@�=�'       ��F	>F�X�t�AI*

training_accuracy�$<?�P~,#       ��wC	��$Y�t�AJ*

training_loss���@~	�'       ��F	�$Y�t�AJ*

training_accuracyn�;?Iڣi#       ��wC	�mY�t�AK*

training_lossS��@����'       ��F	_mY�t�AK*

training_accuracy�m=?����%       �6�	\ӄY�t�AK*

validation_loss2��@�^})       7�_ 	�ԄY�t�AK*

validation_accuracy��H?�0]�#       ��wC	a�Y�t�AL*

training_loss<��@�/�'       ��F	�b�Y�t�AL*

training_accuracyI2=?�"��#       ��wC	�TZ�t�AM*

training_lossd`�@�:{B'       ��F	sVZ�t�AM*

training_accuracy @<?����#       ��wC	��WZ�t�AN*

training_loss�@�=��'       ��F	4�WZ�t�AN*

training_accuracy�>?���#       ��wC	 �Z�t�AO*

training_lossOR�@�9$E'       ��F	j�Z�t�AO*

training_accuracy��=?.�C7#       ��wC	2��Z�t�AP*

training_loss��@��G'       ��F	���Z�t�AP*

training_accuracy�6=?�U�y%       �6�	 [�t�AP*

validation_loss�n�@LQ.)       7�_ 	� [�t�AP*

validation_accuracyI�H?(�H�#       ��wC	D>G[�t�AQ*

training_loss���@�.�'       ��F	�?G[�t�AQ*

training_accuracy�=?R��#       ��wC	���[�t�AR*

training_loss��@�;�'       ��F	 ��[�t�AR*

training_accuracy�??��}#       ��wC	Y��[�t�AS*

training_loss���@�3�'       ��F	���[�t�AS*

training_accuracy��=?d�"�#       ��wC	�\�t�AT*

training_loss ��@U���'       ��F	\�t�AT*

training_accuracy �=?@�ss#       ��wC	�'h\�t�AU*

training_loss���@#�ȼ'       ��F	l)h\�t�AU*

training_accuracy @>?yױ(%       �6�	�\�t�AU*

validation_loss���@�F$g)       7�_ 	��\�t�AU*

validation_accuracy��H?�eT`#       ��wC	�|�\�t�AV*

training_loss���@B��'       ��F	\~�\�t�AV*

training_accuracy��>?��Y7#       ��wC	u�]�t�AW*

training_loss��@�7�v'       ��F	��]�t�AW*

training_accuracyn>?��#       ��wC	
yT]�t�AX*

training_lossq�@����'       ��F	HzT]�t�AX*

training_accuracy��>?�'��#       ��wC	\��]�t�AY*

training_lossk�@��}c'       ��F	;��]�t�AY*

training_accuracyn�=?���#       ��wC	�Q�]�t�AZ*

training_loss���@]�s�'       ��F	�R�]�t�AZ*

training_accuracyn;??X�T%       �6�	� ^�t�AZ*

validation_lossKֳ@�'M?)       7�_ 	< ^�t�AZ*

validation_accuracy��J?{��\#       ��wC	�bL^�t�A[*

training_lossV1�@G���'       ��F	dL^�t�A[*

training_accuracyI�=?�jq#       ��wC	��^�t�A\*

training_loss�C�@D�U'       ��F	=�^�t�A\*

training_accuracy��??%=�u#       ��wC	�}�^�t�A]*

training_loss��@���@'       ��F	-�^�t�A]*

training_accuracy��<?�Lhp#       ��wC	�F"_�t�A^*

training_loss
��@t�v'       ��F	PH"_�t�A^*

training_accuracy%I>?q�##       ��wC	�bn_�t�A_*

training_loss���@�+{'       ��F	dn_�t�A_*

training_accuracy�m??+�$4%       �6�	fP�_�t�A_*

validation_loss�^�@OM�	)       7�_ 	�Q�_�t�A_*

validation_accuracyۖI?\8�#       ��wC	*m�_�t�A`*

training_loss��@���'       ��F	in�_�t�A`*

training_accuracyn�??��'#       ��wC	U�`�t�Aa*

training_loss��@���"'       ��F	��`�t�Aa*

training_accuracy�m??�6��#       ��wC	ɹ\`�t�Ab*

training_loss��@@��*'       ��F	(�\`�t�Ab*

training_accuracyn{@?��#       ��wC	rԧ`�t�Ac*

training_loss�E�@��z'       ��F	�է`�t�Ac*

training_accuracy �??����#       ��wC	X0�`�t�Ad*

training_loss,j�@٧�8'       ��F	�1�`�t�Ad*

training_accuracy�M??3�I=%       �6�	6�a�t�Ad*

validation_loss��@�;|�)       7�_ 	��a�t�Ad*

validation_accuracyn�J? `�#       ��wC	��Oa�t�Ae*

training_lossWN�@�s�.'       ��F	2�Oa�t�Ae*

training_accuracyn{>?��#       ��wC	:��a�t�Af*

training_loss<��@�
�'       ��F	���a�t�Af*

training_accuracy  >?����#       ��wC	��a�t�Ag*

training_loss�/�@ԥċ'       ��F	/�a�t�Ag*

training_accuracyn;??^���#       ��wC	�'b�t�Ah*

training_loss0H�@~8zR'       ��F	w�'b�t�Ah*

training_accuracyn{@?5�"�#       ��wC	�ob�t�Ai*

training_loss��@�C�'       ��F	Z�ob�t�Ai*

training_accuracy�??��a%       �6�	Y�b�t�Ai*

validation_loss>`�@&-)       7�_ 	��b�t�Ai*

validation_accuracyn�K?���#       ��wC	��b�t�Aj*

training_loss��@�N�'       ��F	���b�t�Aj*

training_accuracy%I??v���#       ��wC	\7c�t�Ak*

training_losss�@^Ay'       ��F	�8c�t�Ak*

training_accuracyn�>?����#       ��wC	%�_c�t�Al*

training_loss�@Jw�'       ��F	��_c�t�Al*

training_accuracy%�@?��s#       ��wC	b�c�t�Am*

training_loss/~�@��Լ'       ��F	��c�t�Am*

training_accuracyۖ??����#       ��wC	�c�t�An*

training_lossв�@�=o�'       ��F	�	�c�t�An*

training_accuracy��@?B%       �6�	ed�t�An*

validation_loss���@2ZX)       7�_ 	|fd�t�An*

validation_accuracy �K?��#       ��wC	Md�t�Ao*

training_loss��@�5̄'       ��F	|Md�t�Ao*

training_accuracy�V??]��L#       ��wC	�`�d�t�Ap*

training_loss:0�@��~h'       ��F	ab�d�t�Ap*

training_accuracy�??�1e#       ��wC	�8�d�t�Aq*

training_loss��@�K�'       ��F	(:�d�t�Aq*

training_accuracyn[@?����#       ��wC	�6"e�t�Ar*

training_loss�U�@���Z'       ��F	38"e�t�Ar*

training_accuracy��>?��c#       ��wC	D�ie�t�As*

training_loss�@�@�F�4'       ��F	��ie�t�As*

training_accuracyۖ@?�X�}%       �6�	��e�t�As*

validation_loss���@�-Zx)       7�_ 	K�e�t�As*

validation_accuracy �L?N�|#       ��wC	���e�t�At*

training_loss�'�@x� '       ��F	&��e�t�At*

training_accuracy�??�-#       ��wC	�f�t�Au*

training_loss���@E>�'       ��F	5�f�t�Au*

training_accuracy%�=?*ξ
#       ��wC	c�\f�t�Av*

training_loss���@��P%'       ��F	��\f�t�Av*

training_accuracy `@?re#       ��wC	0}�f�t�Aw*

training_losss�@$���'       ��F	i~�f�t�Aw*

training_accuracy��@?��s�#       ��wC	�
�f�t�Ax*

training_loss��@�$�8'       ��F	�f�t�Ax*

training_accuracyI�@?I+��%       �6�	Υg�t�Ax*

validation_loss��@���:)       7�_ 	�g�t�Ax*

validation_accuracyI�L?n��-#       ��wC	�MKg�t�Ay*

training_loss��@qW�'       ��F	VOKg�t�Ay*

training_accuracy��@?�kf�#       ��wC	גg�t�Az*

training_loss�O�@�dQ�'       ��F	vؒg�t�Az*

training_accuracy%	A?�e�#       ��wC	��g�t�A{*

training_loss|��@`2�='       ��F	� �g�t�A{*

training_accuracy۶@?��k#       ��wC	�|"h�t�A|*

training_lossz�@(K�?'       ��F	�}"h�t�A|*

training_accuracy @A?g��F#       ��wC	7�hh�t�A}*

training_loss:��@
Ni�'       ��F	��hh�t�A}*

training_accuracy�B?灷�%       �6�	�h�t�A}*

validation_lossP߮@L���)       7�_ 	Ăh�t�A}*

validation_accuracy�vM?l6��#       ��wC	9�h�t�A~*

training_loss���@�+'       ��F	r:�h�t�A~*

training_accuracy �@?�Y��#       ��wC	s�i�t�A*

training_loss�q�@�+�B'       ��F	��i�t�A*

training_accuracyۖA?5��$       B+�M	i/Yi�t�A�*

training_lossna�@w��(       �pJ	�0Yi�t�A�*

training_accuracyn{@?Q��$       B+�M	/o�i�t�A�*

training_loss�@�g]J(       �pJ	�p�i�t�A�*

training_accuracynA?4̙�$       B+�M	��i�t�A�*

training_loss]þ@�pa(       �pJ	��i�t�A�*

training_accuracyI�@?�E>&       sO� 	07j�t�A�*

validation_lossT�@*ʀ *       ����	�8j�t�A�*

validation_accuracyI2M?^wI�$       B+�M	OsHj�t�A�*

training_lossF��@��(       �pJ	�tHj�t�A�*

training_accuracy۶A?����$       B+�M	խ�j�t�A�*

training_losso��@���,(       �pJ	S��j�t�A�*

training_accuracy��??n�/$       B+�M	��j�t�A�*

training_loss��@u��%(       �pJ	y��j�t�A�*

training_accuracy �A?v���$       B+�M	P7k�t�A�*

training_lossY�@���(       �pJ	�8k�t�A�*

training_accuracy�A?��yh$       B+�M	,vek�t�A�*

training_loss߀�@QƓF(       �pJ	�wek�t�A�*

training_accuracyI�A?3�M|&       sO� 	pZk�t�A�*

validation_loss�֮@ƛ>�*       ����	�[k�t�A�*

validation_accuracyn�M?�=I$       B+�M	�L�k�t�A�*

training_losse��@���i(       �pJ	�M�k�t�A�*

training_accuracyn�A?]P�|$       B+�M	��l�t�A�*

training_loss�-�@� B�(       �pJ	�l�t�A�*

training_accuracyn�@?�*�$       B+�M	��Tl�t�A�*

training_loss#5�@��L(       �pJ	^�Tl�t�A�*

training_accuracyIrB?+�W!$       B+�M	tÛl�t�A�*

training_loss�+�@�k�q(       �pJ	�ěl�t�A�*

training_accuracy%�B?E[��$       B+�M	7�l�t�A�*

training_lossㄿ@����(       �pJ	��l�t�A�*

training_accuracyI�@?���&       sO� 	�~�l�t�A�*

validation_loss���@�N�*       ����	"��l�t�A�*

validation_accuracy�6N?�2�V$       B+�M	�$Bm�t�A�*

training_losst�@P�5(       �pJ	�%Bm�t�A�*

training_accuracy%iA?�Ǻm$       B+�M	�m�t�A�*

training_loss���@I`�B(       �pJ	��m�t�A�*

training_accuracy�-A?�$       B+�M	���m�t�A�*

training_loss�G�@>'�M(       �pJ	���m�t�A�*

training_accuracy�A?����$       B+�M	[n�t�A�*

training_loss���@��ld(       �pJ	�n�t�A�*

training_accuracy%IC?��G$       B+�M	��^n�t�A�*

training_loss�@ȅr(       �pJ	P�^n�t�A�*

training_accuracyI2@?�K�&       sO� 	|ovn�t�A�*

validation_loss(�@�j;�*       ����	�pvn�t�A�*

validation_accuracy�MN?Q�T$       B+�M	2�n�t�A�*

training_loss��@i���(       �pJ	�3�n�t�A�*

training_accuracy��A?>�t}$       B+�M	@�o�t�A�*

training_loss8��@�s�(       �pJ	��o�t�A�*

training_accuracy �A?+E$       B+�M	�No�t�A�*

training_loss���@��y�(       �pJ	W�No�t�A�*

training_accuracy%�A?i�F�$       B+�M	 �o�t�A�*

training_lossD��@`с(       �pJ	G�o�t�A�*

training_accuracy��B?��kr$       B+�M	B�o�t�A�*

training_lossh�@�t��(       �pJ	IC�o�t�A�*

training_accuracy �B?�0�&       sO� 	E�o�t�A�*

validation_loss�t�@ ���*       ����	��o�t�A�*

validation_accuracy��N?�S��$       B+�M	h8<p�t�A�*

training_lossM��@����(       �pJ	�9<p�t�A�*

training_accuracyn;B?�~`$       B+�M	�k�p�t�A�*

training_loss^��@���(       �pJ	em�p�t�A�*

training_accuracy%�B?~���$       B+�M	#��p�t�A�*

training_loss���@�Ў)(       �pJ	w��p�t�A�*

training_accuracyn�B?�}��$       B+�M	��q�t�A�*

training_loss�¹@
a�(       �pJ	�q�t�A�*

training_accuracyn�B?F|A$       B+�M	'_q�t�A�*

training_lossNL�@)p2(       �pJ	O(_q�t�A�*

training_accuracyI�A?�<�&       sO� 	�%wq�t�A�*

validation_loss?��@��2*       ����	/'wq�t�A�*

validation_accuracyn�N?�X�$       B+�M	d�q�t�A�*

training_loss���@~R۽(       �pJ	~e�q�t�A�*

training_accuracyI�A?WcM$       B+�M	��r�t�A�*

training_loss�J�@�'\�(       �pJ	8�r�t�A�*

training_accuracy�$C?s�W�$       B+�M	P�Qr�t�A�*

training_loss%Ϲ@���(       �pJ	ƏQr�t�A�*

training_accuracy%iB?�TN�$       B+�M	�r�t�A�*

training_lossvȺ@�_|�(       �pJ	Bėr�t�A�*

training_accuracy%�@?�"}�$       B+�M	���r�t�A�*

training_loss�\�@> c(       �pJ	Z��r�t�A�*

training_accuracyIRC?#�@&       sO� 	�+�r�t�A�*

validation_loss��@�\��*       ����	�,�r�t�A�*

validation_accuracy�$P?SX��$       B+�M	^�@s�t�A�*

training_lossJ�@P,�(       �pJ	��@s�t�A�*

training_accuracyn�B?@�xB$       B+�M	��s�t�A�*

training_loss��@��`�(       �pJ	��s�t�A�*

training_accuracy �B?�Ѵ#$       B+�M	�}�s�t�A�*

training_loss`@�@�yF�(       �pJ	E�s�t�A�*

training_accuracyn�B?��s$       B+�M	 �t�t�A�*

training_lossTb�@��-D(       �pJ	m�t�t�A�*

training_accuracy�vC?�H�$       B+�M	̧at�t�A�*

training_lossb&�@���(       �pJ	�at�t�A�*

training_accuracy%)C?����&       sO� 	�Eyt�t�A�*

validation_loss!_�@�.C�*       ����	Gyt�t�A�*

validation_accuracy%�P?�~Z$       B+�M	-��t�t�A�*

training_loss2�@�VҀ(       �pJ	n��t�t�A�*

training_accuracy%iC?Q	-$       B+�M	2�	u�t�A�*

training_loss�ض@�7�\(       �pJ	��	u�t�A�*

training_accuracy�6D?��G$       B+�M	w�Su�t�A�*

training_loss͊�@_�us(       �pJ	�Su�t�A�*

training_accuracynD?݂^i$       B+�M	��u�t�A�*

training_loss�ʷ@O�i(       �pJ	{��u�t�A�*

training_accuracy��C?�"C�$       B+�M	SK�u�t�A�*

training_loss���@�n�I(       �pJ	�L�u�t�A�*

training_accuracyI�C?Q�&       sO� 	�0v�t�A�*

validation_loss06�@Jg	H*       ����	)2v�t�A�*

validation_accuracyIRP?���$       B+�M	Kv�t�A�*

training_loss.�@t�(       �pJ	~Kv�t�A�*

training_accuracy۶D?<I�U$       B+�M	ڞ�v�t�A�*

training_lossCp�@��H�(       �pJ	>��v�t�A�*

training_accuracy  C?�8��$       B+�M	΅�v�t�A�*

training_loss�ʵ@��jT(       �pJ	��v�t�A�*

training_accuracy��B?�~~O$       B+�M	�?!w�t�A�*

training_loss¤�@�ur(       �pJ	�@!w�t�A�*

training_accuracy۶D?N0{$       B+�M	/�lw�t�A�*

training_loss`C�@�;̇(       �pJ	��lw�t�A�*

training_accuracy�MD?k�*&       sO� 	G�w�t�A�*

validation_loss�%�@��*       ����	��w�t�A�*

validation_accuracy%	Q?��9�$       B+�M	ڕ�w�t�A�*

training_loss� �@>��-(       �pJ	2��w�t�A�*

training_accuracy�C?�_�-$       B+�M	�x�t�A�*

training_loss���@˟��(       �pJ	8x�t�A�*

training_accuracy @E?�C|$       B+�M	��^x�t�A�*

training_loss�@��(       �pJ	��^x�t�A�*

training_accuracy�B?��{$$       B+�M	��x�t�A�*

training_loss*�@�x(       �pJ	\�x�t�A�*

training_accuracy �F?6˞$       B+�M	�d�x�t�A�*

training_lossb\�@�=�h(       �pJ	'f�x�t�A�*

training_accuracy��E?�O/�&       sO� 	e�y�t�A�*

validation_loss���@� �*       ����	՚y�t�A�*

validation_accuracyn�S?l�!$       B+�M	��My�t�A�*

training_loss|�@n1�(       �pJ	�My�t�A�*

training_accuracy�mD?Ji�)$       B+�M	:��y�t�A�*

training_lossD�@Q�(n(       �pJ	��y�t�A�*

training_accuracy۶D?y~9
$       B+�M	���y�t�A�*

training_lossZ>�@A+A(       �pJ	2��y�t�A�*

training_accuracyI�D?m��$       B+�M	��#z�t�A�*

training_loss�ݵ@�]/(       �pJ	��#z�t�A�*

training_accuracyn�D?|���$       B+�M	�mz�t�A�*

training_loss�ò@�v%/(       �pJ	�mz�t�A�*

training_accuracyn�C?�>�s&       sO� 	p��z�t�A�*

validation_loss��@>��*       ����	��z�t�A�*

validation_accuracy�VR?1���$       B+�M	��z�t�A�*

training_loss�@3��(       �pJ	-��z�t�A�*

training_accuracy%�D?m3&P$       B+�M	��{�t�A�*

training_loss�5�@}�8(       �pJ	k�{�t�A�*

training_accuracy��D?i0�c$       B+�M	��a{�t�A�*

training_loss���@���(       �pJ	��a{�t�A�*

training_accuracy�mE?���+$       B+�M	���{�t�A�*

training_loss]z�@ {�c(       �pJ	M��{�t�A�*

training_accuracy �E?�<w�$       B+�M	o��{�t�A�*

training_lossM��@[�R1(       �pJ	���{�t�A�*

training_accuracy�$F?�e�V&       sO� 	nz|�t�A�*

validation_lossh�@}�s�*       ����	�{|�t�A�*

validation_accuracy��S?h8��$       B+�M	AV|�t�A�*

training_loss�ױ@�`�-(       �pJ	DBV|�t�A�*

training_accuracy%�C?cd�1$       B+�M	x>�|�t�A�*

training_loss�G�@����(       �pJ	�?�|�t�A�*

training_accuracy%IF?�~�$       B+�M	�A�|�t�A�*

training_loss\:�@�D�(       �pJ	�C�|�t�A�*

training_accuracy�E?$       B+�M	��,}�t�A�*

training_loss:�@��G(       �pJ	�,}�t�A�*

training_accuracy��D?ڊf�$       B+�M	�;v}�t�A�*

training_loss�ֱ@����(       �pJ	t=v}�t�A�*

training_accuracy%iF?�+6�&       sO� 	8$�}�t�A�*

validation_lossQz�@���W*       ����	�%�}�t�A�*

validation_accuracy�vT?���L$       B+�M	�}�t�A�*

training_loss�Y�@���z(       �pJ	��}�t�A�*

training_accuracyn�D?�[�$       B+�M	��~�t�A�*

training_loss^��@i
�4(       �pJ	V�~�t�A�*

training_accuracyI�E?��ױ$       B+�M	��f~�t�A�*

training_loss�ͱ@c�'+(       �pJ	��f~�t�A�*

training_accuracy�ME?q�x$       B+�M	 �~�t�A�*

training_loss,�@Z<�(       �pJ	h�~�t�A�*

training_accuracy @D?a馏$       B+�M	��~�t�A�*

training_losseH�@��(       �pJ	�~�t�A�*

training_accuracy��C?{�T&       sO� 	#W�t�A�*

validation_lossE£@�Ks�*       ����	lX�t�A�*

validation_accuracy%�S?,i��$       B+�M	��V�t�A�*

training_loss<�@KA��(       �pJ	�V�t�A�*

training_accuracy @E?�w$       B+�M	2���t�A�*

training_loss���@W��(       �pJ	����t�A�*

training_accuracy%�E?B��$       B+�M	�w��t�A�*

training_loss���@U�p�(       �pJ	y��t�A�*

training_accuracy�$E?.�/$       B+�M	`v,��t�A�*

training_loss+��@�V6(       �pJ	�w,��t�A�*

training_accuracyn�D?��w$       B+�M	�Rs��t�A�*

training_loss��@W��(       �pJ	*Ts��t�A�*

training_accuracy�mF?8�B&       sO� 	ht���t�A�*

validation_loss���@��Ӱ*       ����	�u���t�A�*

validation_accuracy%�T?�\$       B+�M	rӀ�t�A�*

training_lossȣ�@j5
(       �pJ	�Ӏ�t�A�*

training_accuracyn�E?�?�D$       B+�M	s0��t�A�*

training_loss6�@P���(       �pJ	�1��t�A�*

training_accuracyIF?mz`�$       B+�M	�Kb��t�A�*

training_loss�	�@.��(       �pJ	DMb��t�A�*

training_accuracy�6E?2��f$       B+�M	8����t�A�*

training_loss��@0�I.(       �pJ	�����t�A�*

training_accuracyn{F?�K�$       B+�M	F���t�A�*

training_loss9Z�@R��>(       �pJ	����t�A�*

training_accuracyI2E?��&&       sO� 	]��t�A�*

validation_loss@D�@�j=*       ����	�^��t�A�*

validation_accuracy�V?�U�$       B+�M	o�W��t�A�*

training_loss�P�@'�u�(       �pJ	ߪW��t�A�*

training_accuracy��F?p��3$       B+�M	�o���t�A�*

training_loss��@z�(       �pJ	�q���t�A�*

training_accuracyn[E?�ٔ�$       B+�M	Ƣ��t�A�*

training_loss���@cl�(       �pJ	���t�A�*

training_accuracy�mE?
���$       B+�M	S:��t�A�*

training_loss�?�@��L=(       �pJ	�:��t�A�*

training_accuracy%�E?���$       B+�M	����t�A�*

training_loss��@����(       �pJ	G���t�A�*

training_accuracyI�D?�;8�&       sO� 	0����t�A�*

validation_loss�Ǡ@<u�z*       ����	f����t�A�*

validation_accuracy�-V?�Z�$       B+�M	�h��t�A�*

training_loss���@���6(       �pJ	�i��t�A�*

training_accuracy��E?Մ$�$       B+�M	� 0��t�A�*

training_lossu��@=��(       �pJ	0��t�A�*

training_accuracyۖF?�}��$       B+�M	�{~��t�A�*

training_loss_�@r��W(       �pJ	V}~��t�A�*

training_accuracyIRE?��IP$       B+�M	��Ǆ�t�A�*

training_loss�ծ@"�(�(       �pJ	��Ǆ�t�A�*

training_accuracyI�E?�9y$       B+�M	۹��t�A�*

training_loss�@��(       �pJ	+���t�A�*

training_accuracyIrF?���m&       sO� 	+�.��t�A�*

validation_loss�0�@�}s�*       ����	a�.��t�A�*

validation_accuracyIX?/vڢ$       B+�M	"&x��t�A�*

training_loss[��@��?(       �pJ	�'x��t�A�*

training_accuracyI�E?�0$       B+�M	� Å�t�A�*

training_loss6��@�~��(       �pJ	�Å�t�A�*

training_accuracy �E?��D�$       B+�M	����t�A�*

training_loss�ެ@Z�P(       �pJ	���t�A�*

training_accuracy�vE?���$       B+�M	��W��t�A�*

training_loss��@g9C�(       �pJ	�W��t�A�*

training_accuracyn[F?�	��$       B+�M	�f���t�A�*

training_loss���@�-�u(       �pJ	�g���t�A�*

training_accuracyn�F?�b�&       sO� 	�9���t�A�*

validation_loss�!�@�x��*       ����	v;���t�A�*

validation_accuracyIRX?�99�$       B+�M	�	��t�A�*

training_loss>��@�ϖ�(       �pJ	��	��t�A�*

training_accuracy �F?>E��$       B+�M	y�T��t�A�*

training_loss6��@Rb}�(       �pJ	N�T��t�A�*

training_accuracy �E?v�@�$       B+�M	e"���t�A�*

training_loss;٫@qq I(       �pJ	$���t�A�*

training_accuracy��G?*�I$       B+�M	����t�A�*

training_loss���@��+�(       �pJ	U���t�A�*

training_accuracyI2G?/�$J$       B+�M	�:��t�A�*

training_loss��@�W�(       �pJ	;:��t�A�*

training_accuracy @F?�B�H&       sO� 	�,T��t�A�*

validation_loss@=�{*       ����	;.T��t�A�*

validation_accuracy�X?xӐ$       B+�M	a���t�A�*

training_lossɂ�@��p(       �pJ	�b���t�A�*

training_accuracy��E?�bV$       B+�M	����t�A�*

training_loss{�@�[(       �pJ	Ǐ��t�A�*

training_accuracyI�D?�ǻp$       B+�M	�1��t�A�*

training_loss�&�@�#(       �pJ	 �1��t�A�*

training_accuracy%�G?�,�\$       B+�M	��x��t�A�*

training_loss���@��u(       �pJ	�x��t�A�*

training_accuracy�DF?B�$       B+�M	Կ��t�A�*

training_loss���@�gQ�(       �pJ	kտ��t�A�*

training_accuracy��E?ŧ�&       sO� 	x׉�t�A�*

validation_lossI|�@���*       ����	� ׉�t�A�*

validation_accuracy �X?q+9�$       B+�M	���t�A�*

training_loss���@7��((       �pJ	R���t�A�*

training_accuracy%�F?>��$       B+�M	��e��t�A�*

training_lossPݫ@z�I(       �pJ	�e��t�A�*

training_accuracy۶F?�*E�$       B+�M	����t�A�*

training_loss�W�@b&m(       �pJ	u����t�A�*

training_accuracyn�F?2R��$       B+�M	���t�A�*

training_loss
t�@��~(       �pJ	����t�A�*

training_accuracyn;F?M�Y�$       B+�M	[<��t�A�*

training_loss���@{��l(       �pJ	|\<��t�A�*

training_accuracyn;G?�]��&       sO� 	,zU��t�A�*

validation_loss�@ҝ��*       ����	�{U��t�A�*

validation_accuracy `X?}�x$       B+�M	�S���t�A�*

training_loss2S�@\Gا(       �pJ	MU���t�A�*

training_accuracyn�F?8�8~$       B+�M	&���t�A�*

training_loss��@�/o(       �pJ	i���t�A�*

training_accuracy�mG?K�s�$       B+�M	��+��t�A�*

training_lossmk�@áDh(       �pJ	7�+��t�A�*

training_accuracy%IG?	�$       B+�M	ӈr��t�A�*

training_loss���@���(       �pJ	:�r��t�A�*

training_accuracy�VG?W���$       B+�M	�����t�A�*

training_loss��@��+(       �pJ	�����t�A�*

training_accuracy�G?��9A&       sO� 	p�Ќ�t�A�*

validation_lossfߛ@$�W�*       ����	��Ќ�t�A�*

validation_accuracy  Y?��%�$       B+�M	*+��t�A�*

training_loss�
�@�� (       �pJ	�,��t�A�*

training_accuracy�vG?�'��$       B+�M	�J`��t�A�*

training_loss�X�@ 3�(       �pJ	L`��t�A�*

training_accuracy��G?*$5$       B+�M	�=���t�A�*

training_loss���@w&Տ(       �pJ	^?���t�A�*

training_accuracy��F?���$       B+�M	����t�A�*

training_loss&~�@�5LK(       �pJ	����t�A�*

training_accuracyIrG?k!�$       B+�M	�5��t�A�*

training_lossdM�@�ɑ(       �pJ	U�5��t�A�*

training_accuracyn[F?�爗&       sO� 	��N��t�A�*

validation_loss�@f!D�*       ����	'�N��t�A�*

validation_accuracy��X?�h�	$       B+�M	1����t�A�*

training_loss$l�@6�c(       �pJ	�����t�A�*

training_accuracyn�H?J �)$       B+�M	��܎�t�A�*

training_lossJ��@�:(       �pJ	<�܎�t�A�*

training_accuracy  H?��$$       B+�M	�"��t�A�*

training_loss��@�s>!(       �pJ	�"��t�A�*

training_accuracy%)G?���$       B+�M	�Yi��t�A�*

training_loss�d�@R�(       �pJ	#[i��t�A�*

training_accuracy��G?'���$       B+�M	!����t�A�*

training_lossJ��@`n$(       �pJ	�����t�A�*

training_accuracyI�G?Ӆ&       sO� 	�Ǐ�t�A�*

validation_loss�O�@F*       ����	�Ǐ�t�A�*

validation_accuracy�vY?r�W�$       B+�M	0C��t�A�*

training_loss���@����(       �pJ	�D��t�A�*

training_accuracy�$G?�y�]$       B+�M	�TW��t�A�*

training_losst�@?�f�(       �pJ	:VW��t�A�*

training_accuracy @I?�k0�$       B+�M	����t�A�*

training_loss4֧@��T�(       �pJ	l����t�A�*

training_accuracy%)H?�ܛ�$       B+�M	����t�A�*

training_loss�-�@�e%(       �pJ	5���t�A�*

training_accuracy��G?5Y�C$       B+�M	^s.��t�A�*

training_loss���@舘�(       �pJ	�t.��t�A�*

training_accuracyn�G?��&       sO� 	��G��t�A�*

validation_loss�@�2�c*       ����	�G��t�A�*

validation_accuracy�MY?�,u�$       B+�M	�'���t�A�*

training_loss��@N4��(       �pJ	)���t�A�*

training_accuracy��G?��J�$       B+�M	��Ց�t�A�*

training_loss++�@��`�(       �pJ	$�Ց�t�A�*

training_accuracy�dH?��$       B+�M	A��t�A�*

training_loss]&�@u�(       �pJ	���t�A�*

training_accuracyn{H?�b��$       B+�M	&�c��t�A�*

training_loss���@��Ғ(       �pJ	n�c��t�A�*

training_accuracyn;H?�'�$       B+�M	,����t�A�*

training_loss4��@��Κ(       �pJ	v����t�A�*

training_accuracy�DG?&V&       sO� 	2h�t�A�*

validation_loss���@N(�*       ����	pi�t�A�*

validation_accuracyn[Z? W�Y$       B+�M	���t�A�*

training_loss��@,	eu(       �pJ	C���t�A�*

training_accuracy  H?�\�{$       B+�M	lS��t�A�*

training_lossDW�@g�W(       �pJ	NmS��t�A�*

training_accuracyn{G?s5�$       B+�M	v���t�A�*

training_loss�j�@���(       �pJ	����t�A�*

training_accuracyn�H?r!��$       B+�M	wJ��t�A�*

training_loss��@T�"(       �pJ	�K��t�A�*

training_accuracy%�H?R��$       B+�M	0�*��t�A�*

training_loss��@m�0(       �pJ	��*��t�A�*

training_accuracy��H?U���&       sO� 	�iB��t�A�*

validation_lossҐ�@�*<*       ����	1kB��t�A�*

validation_accuracy��Y?��h�$       B+�M	cJ���t�A�*

training_loss���@{���(       �pJ	�K���t�A�*

training_accuracy�H?��R$       B+�M	��Ҕ�t�A�*

training_loss	��@�tC�(       �pJ	%�Ҕ�t�A�*

training_accuracyn;I?
��}$       B+�M	����t�A�*

training_loss.k�@<��X(       �pJ	���t�A�*

training_accuracy%�H? ��I$       B+�M	Z|a��t�A�*

training_loss^�@���7(       �pJ	�}a��t�A�*

training_accuracy `H?k��$       B+�M	Ӝ���t�A�*

training_loss%�@�t��(       �pJ	E����t�A�*

training_accuracyn{G?��b�&       sO� 	�����t�A�*

validation_loss$H�@��߼*       ����	5����t�A�*

validation_accuracy�vY?���$       B+�M	^���t�A�*

training_loss�m�@oh�(       �pJ	����t�A�*

training_accuracy��H?�P�$       B+�M	�_Q��t�A�*

training_lossiW�@"J�I(       �pJ	aQ��t�A�*

training_accuracy%�F?���$       B+�M	�����t�A�*

training_loss*Y�@ƭ�q(       �pJ	�����t�A�*

training_accuracy @H?�7�$       B+�M	>O��t�A�*

training_loss=T�@����(       �pJ	�P��t�A�*

training_accuracy%�H?PrN\$       B+�M	W'��t�A�*

training_loss$@�@XJR(       �pJ	�'��t�A�*

training_accuracy��H?�e�&       sO� 	�w>��t�A�*

validation_lossw�@"q%!*       ����	9y>��t�A�*

validation_accuracy%)Z?-�!v$       B+�M	�����t�A�*

training_lossh�@|�(       �pJ	 ����t�A�*

training_accuracy��G?�7e$       B+�M	D�̗�t�A�*

training_loss<5�@2�(       �pJ	��̗�t�A�*

training_accuracy �H?J6��$       B+�M	܂��t�A�*

training_loss�L�@�t�(       �pJ	D���t�A�*

training_accuracy��G?��2�$       B+�M	��Z��t�A�*

training_loss���@[� �(       �pJ	-�Z��t�A�*

training_accuracy��H?��$       B+�M	�����t�A�*

training_loss���@s�]�(       �pJ	����t�A�*

training_accuracyIrH?�޵�&       sO� 	丘�t�A�*

validation_loss�_�@�+s�*       ����	}帘�t�A�*

validation_accuracyn�Y?��I$       B+�M	F����t�A�*

training_losst��@Dn�D(       �pJ	�����t�A�*

training_accuracy%	J?���$       B+�M	�=G��t�A�*

training_loss!:�@���a(       �pJ	.?G��t�A�*

training_accuracy%�H?]�Cl$       B+�M	�����t�A�*

training_loss�ɡ@���(       �pJ	ٰ���t�A�*

training_accuracy�I?��&	$       B+�M	�kԙ�t�A�*

training_loss"��@�`š(       �pJ	mԙ�t�A�*

training_accuracy  H?�>$       B+�M	}���t�A�*

training_loss��@�3=(       �pJ	Č��t�A�*

training_accuracy۶I?�1`�&       sO� 	~�4��t�A�*

validation_loss۬�@!צ*       ����	��4��t�A�*

validation_accuracyI�Y?��g�$       B+�M	~}��t�A�*

training_lossm6�@�p�5(       �pJ	�}��t�A�*

training_accuracyn�H?��$       B+�M	oĚ�t�A�*

training_loss!<�@�K/(       �pJ	opĚ�t�A�*

training_accuracyIH?$	�$       B+�M	�;��t�A�*

training_lossgX�@���F(       �pJ	J=��t�A�*

training_accuracyI�H?nw@$       B+�M	TT��t�A�*

training_lossu��@M/i.(       �pJ	xUT��t�A�*

training_accuracy �G?�T�/$       B+�M	PA���t�A�*

training_loss�O�@��ʼ(       �pJ	�B���t�A�*

training_accuracyn�H?ɵX@&       sO� 	W����t�A�*

validation_loss�'�@� �*       ����	Ǐ���t�A�*

validation_accuracy @Z?�B�!$       B+�M	3����t�A�*

training_losss��@'��3(       �pJ	�����t�A�*

training_accuracy��I?�I-�$       B+�M	��?��t�A�*

training_loss��@1��(       �pJ	��?��t�A�*

training_accuracy��G?��$       B+�M	#���t�A�*

training_loss�p�@�F(       �pJ	����t�A�*

training_accuracy @I?�.�
$       B+�M	V͜�t�A�*

training_loss-O�@��>�(       �pJ	�͜�t�A�*

training_accuracyIRJ?�`y<$       B+�M	����t�A�*

training_loss���@X��r(       �pJ	���t�A�*

training_accuracy%	J?c�dj&       sO� 	��,��t�A�*

validation_lossJ��@o�i�*       ����	X�,��t�A�*

validation_accuracyn�Y?C�<�$       B+�M	�w��t�A�*

training_loss+�@���(       �pJ	]�w��t�A�*

training_accuracyII?�� d$       B+�M	"	���t�A�*

training_loss�ޣ@*f�>(       �pJ	h
���t�A�*

training_accuracy۶G?
 -�$       B+�M	�f��t�A�*

training_loss���@�n�(       �pJ	h��t�A�*

training_accuracy۶H?mP��$       B+�M	��O��t�A�*

training_loss���@�t�(       �pJ	��O��t�A�*

training_accuracy�H?Zȭ�$       B+�M	8斞�t�A�*

training_losskƤ@v��((       �pJ	v疞�t�A�*

training_accuracyn�G?��מ&       sO� 	�Ů��t�A�*

validation_loss�F�@�0A*       ����	.Ǯ��t�A�*

validation_accuracy �Z?�@4�$       B+�M	uD���t�A�*

training_lossF7�@�(�(       �pJ	�E���t�A�*

training_accuracy%�I?��[6$       B+�M	�<��t�A�*

training_loss�*�@�Zt�(       �pJ	X<��t�A�*

training_accuracy%�I?h}��$       B+�M	Pc���t�A�*

training_loss�{�@U�J�(       �pJ	�d���t�A�*

training_accuracy�MH?����$       B+�M	j˟�t�A�*

training_loss�c�@@�:(       �pJ	�˟�t�A�*

training_accuracyIRI?��+�$       B+�M	����t�A�*

training_loss[��@oǎ�(       �pJ	n���t�A�*

training_accuracy��I?*��&       sO� 	�3*��t�A�*

validation_lossM�@βa*       ����	
5*��t�A�*

validation_accuracy�[?�@$       B+�M	��q��t�A�*

training_loss���@�5k�(       �pJ	C�q��t�A�*

training_accuracyIRI?�C�<$       B+�M	�����t�A�*

training_loss~�@�z��(       �pJ	􁺠�t�A�*

training_accuracyn�H?bQ�g$       B+�M	���t�A�*

training_lossMK�@ �)(       �pJ	e ��t�A�*

training_accuracy��H?�qL1$       B+�M	�TK��t�A�*

training_loss�z�@���(       �pJ	�UK��t�A�*

training_accuracyIrI?���$       B+�M	�����t�A�*

training_loss���@k2�(       �pJ	�����t�A�*

training_accuracy%IH?�|&       sO� 	r����t�A�*

validation_loss��@�*!5*       ����	�����t�A�*

validation_accuracy�vZ?��$       B+�M	�Q��t�A�*

training_loss巠@��BU(       �pJ	 S��t�A�*

training_accuracy�DI?,��$       B+�M	Ư8��t�A�*

training_loss���@����(       �pJ	,�8��t�A�*

training_accuracy�VI?�s�$       B+�M	)X���t�A�*

training_loss�q�@+8s�(       �pJ	qY���t�A�*

training_accuracy�H?B�8�$       B+�M	3QȢ�t�A�*

training_loss�@Y�y
(       �pJ	�RȢ�t�A�*

training_accuracy `H?e�Xp$       B+�M	_���t�A�*

training_lossݥ�@���.(       �pJ	΁��t�A�*

training_accuracyn;I?��=&       sO� 	��(��t�A�*

validation_loss*��@�Z�*       ����	?�(��t�A�*

validation_accuracy �Z? ���$       B+�M	�o��t�A�*

training_lossMc�@2L�(       �pJ	Uo��t�A�*

training_accuracyIRL?j/$       B+�M	�o���t�A�*

training_lossb��@ӑRT(       �pJ	Yq���t�A�*

training_accuracyn�H?O�t$       B+�M	�~���t�A�*

training_loss�@S��(       �pJ	,����t�A�*

training_accuracy��I?�7�$       B+�M	��F��t�A�*

training_loss�v�@x�ܠ(       �pJ	#�F��t�A�*

training_accuracy �I?���$       B+�M	js���t�A�*

training_loss���@\��(       �pJ	�t���t�A�*

training_accuracy �H?�%�Z&       sO� 	ﬣ��t�A�*

validation_loss�O�@35G*       ����	a����t�A�*

validation_accuracy @[?�
p$       B+�M	<Z��t�A�*

training_loss,��@�b�j(       �pJ	�[��t�A�*

training_accuracyI�I?�{�$       B+�M	��2��t�A�*

training_loss@Ξ@����(       �pJ	33��t�A�*

training_accuracyn�I?��V$       B+�M	x{��t�A�*

training_loss�<�@�v(       �pJ	�y{��t�A�*

training_accuracy�I?��K�$       B+�M	?�å�t�A�*

training_lossA��@�#+�(       �pJ	��å�t�A�*

training_accuracy%iI?�;3�$       B+�M	 o��t�A�*

training_loss�;�@�V��(       �pJ	ip��t�A�*

training_accuracy�-I?�B�&       sO� 	w%��t�A�*

validation_lossH��@��*       ����	�%��t�A�*

validation_accuracy�m[?7\�d$       B+�M	Q	n��t�A�*

training_loss�{�@y`(       �pJ	�
n��t�A�*

training_accuracy��I?�o&}$       B+�M	~ķ��t�A�*

training_loss��@��ց(       �pJ	�ŷ��t�A�*

training_accuracy��I?r�F�$       B+�M	�����t�A�*

training_lossߞ@�`?�(       �pJ	D����t�A�*

training_accuracy��I?�r�$       B+�M	* H��t�A�*

training_loss�@ny��(       �pJ	�H��t�A�*

training_accuracyۖJ?Y��$       B+�M	9K���t�A�*

training_loss�.�@,ZuI(       �pJ	�L���t�A�*

training_accuracyI�J?eD�&       sO� 	����t�A�*

validation_loss�@,�P*       ����	����t�A�*

validation_accuracy%)Z?/H�$       B+�M	���t�A�*

training_loss��@�Bj�(       �pJ	� ��t�A�*

training_accuracyIRI?0GI$       B+�M	Y�7��t�A�*

training_loss蚟@'��)(       �pJ	��7��t�A�*

training_accuracyIrI?Ǯq`$       B+�M	VŁ��t�A�*

training_lossݞ@=�(       �pJ	�Ɓ��t�A�*

training_accuracyIrI?���$       B+�M	 Rɨ�t�A�*

training_loss?��@��p�(       �pJ	eSɨ�t�A�*

training_accuracy �H?�T͞$       B+�M	Y���t�A�*

training_loss���@�WUc(       �pJ	ː��t�A�*

training_accuracyn�I?����&       sO� 	b*��t�A�*

validation_loss���@6}�*       ����	�*��t�A�*

validation_accuracyIr[?qH��$       B+�M	�p��t�A�*

training_loss{��@=�_�(       �pJ	��p��t�A�*

training_accuracy �I?���$       B+�M	6η��t�A�*

training_loss�|�@6IF4(       �pJ	�Ϸ��t�A�*

training_accuracy�6J?�/�$       B+�M	*����t�A�*

training_loss͞@so�(       �pJ	�����t�A�*

training_accuracyn�H?��s$       B+�M	@G��t�A�*

training_loss��@�t(       �pJ	hAG��t�A�*

training_accuracy�VK?����$       B+�M	 ����t�A�*

training_lossc��@k��g(       �pJ	�����t�A�*

training_accuracy �J?t&       sO� 	7���t�A�*

validation_loss�z�@���A*       ����	<8���t�A�*

validation_accuracy��\?�.s$       B+�M	S���t�A�*

training_lossr �@Y(       �pJ	ܕ��t�A�*

training_accuracy�ML?Z�$       B+�M	:�3��t�A�*

training_loss@yX�%(       �pJ	��3��t�A�*

training_accuracy%�H?��$       B+�M	�z��t�A�*

training_loss��@��?�(       �pJ	]	z��t�A�*

training_accuracy��I?��
%$       B+�M	�«�t�A�*

training_loss�u�@�j(       �pJ	«�t�A�*

training_accuracy%�I?p#9�$       B+�M	b	��t�A�*

training_lossk/�@�$�(       �pJ	�	��t�A�*

training_accuracy%�I?���&       sO� 	�1"��t�A�*

validation_loss�3�@=�	�*       ����	
3"��t�A�*

validation_accuracy�]?�s+�$       B+�M	��h��t�A�*

training_lossќ@��:�(       �pJ	�h��t�A�*

training_accuracy�K?�+��$       B+�M	lǯ��t�A�*

training_loss�@cP�(       �pJ	�ȯ��t�A�*

training_accuracy��I?��� $       B+�M	�7���t�A�*

training_loss�H�@��2l(       �pJ	@9���t�A�*

training_accuracy�mK?Ǫa�$       B+�M	�!?��t�A�*

training_loss�ʞ@����(       �pJ	�"?��t�A�*

training_accuracy��I?�l�$       B+�M	����t�A�*

training_lossE�@J�h�(       �pJ	���t�A�*

training_accuracy�6I?B3�&       sO� 	�����t�A�*

validation_loss��@�E�*       ����	f����t�A�*

validation_accuracy��\?����$       B+�M	�.��t�A�*

training_loss�t�@�y��(       �pJ	*0��t�A�*

training_accuracy�I?���$       B+�M	-$.��t�A�*

training_loss��@���(       �pJ	�%.��t�A�*

training_accuracy @J?���o$       B+�M	��t��t�A�*

training_loss0c�@q �(       �pJ	$�t��t�A�*

training_accuracy �H?7Y$       B+�M	w���t�A�*

training_loss,{�@�	�(       �pJ	����t�A�*

training_accuracyn;J?����$       B+�M	�P��t�A�*

training_loss_�@p�My(       �pJ	~R��t�A�*

training_accuracy%	I?X�3�&       sO� 	����t�A�*

validation_loss��@���*       ����	K���t�A�*

validation_accuracy��[?��$       B+�M	�c��t�A�*

training_loss�W�@����(       �pJ	8�c��t�A�*

training_accuracy%�I?�+B$       B+�M	#���t�A�*

training_lossY��@f�L�(       �pJ	����t�A�*

training_accuracynJ?�*�$       B+�M	����t�A�*

training_loss=�@ǳ�b(       �pJ	����t�A�*

training_accuracynK?�m�$       B+�M	2�;��t�A�*

training_loss�@�+ol(       �pJ	l�;��t�A�*

training_accuracy�MH?%�z�$       B+�M	�$���t�A�*

training_lossS%�@i'.�(       �pJ	/&���t�A�*

training_accuracy�vK?`Ky�&       sO� 	!ڜ��t�A�*

validation_lossS��@R'�*       ����	bۜ��t�A�*

validation_accuracy%)]?X0͸$       B+�M	�*��t�A�*

training_loss1ĝ@��c (       �pJ	,��t�A�*

training_accuracy��H?	�^<$       B+�M	1)��t�A�*

training_loss`f�@,˒^(       �pJ	�)��t�A�*

training_accuracy��I?j��$       B+�M	�-p��t�A�*

training_loss�n�@͟�g(       �pJ	N/p��t�A�*

training_accuracy�VJ? ��$       B+�M	x]���t�A�*

training_loss��@9�N6(       �pJ	�^���t�A�*

training_accuracy�MK?Zk��$       B+�M	J ��t�A�*

training_lossd<�@����(       �pJ	�K ��t�A�*

training_accuracy�L?P�2�&       sO� 	����t�A�*

validation_loss4�@.YG�*       ����	f���t�A�*

validation_accuracy��\?�J��$       B+�M	��`��t�A�*

training_lossi6�@}��C(       �pJ	�`��t�A�*

training_accuracy�$J?j#D�$       B+�M	'ӧ��t�A�*

training_loss�$�@+���(       �pJ	�ԧ��t�A�*

training_accuracy �I?��F�$       B+�M	!Q��t�A�*

training_loss��@}�+�(       �pJ	ZR��t�A�*

training_accuracy�VJ?M�K�$       B+�M	07��t�A�*

training_loss@&%�'(       �pJ	\17��t�A�*

training_accuracy��J?h!�M$       B+�M	�~��t�A�*

training_loss�I�@�윔(       �pJ	.�~��t�A�*

training_accuracy�$K?{��&       sO� 	$z���t�A�*

validation_loss�v�@� K�*       ����	e{���t�A�*

validation_accuracyn�]?���$       B+�M	�߳�t�A�*

training_loss���@��(       �pJ	t�߳�t�A�*

training_accuracyI�J?�ej$       B+�M	'��t�A�*

training_loss�F�@}X��(       �pJ	y'��t�A�*

training_accuracyIrI?z��$       B+�M	��l��t�A�*

training_lossm��@ �|�(       �pJ	 �l��t�A�*

training_accuracy�dJ?�&J$       B+�M	d>���t�A�*

training_loss���@��"(       �pJ	�?���t�A�*

training_accuracy%�I?s�$       B+�M	>z���t�A�*

training_loss��@K<&(       �pJ	�{���t�A�*

training_accuracy��I?���&       sO� 	LQ��t�A�*

validation_loss���@VR&�*       ����	�R��t�A�*

validation_accuracyI�]?��`$       B+�M	2�]��t�A�*

training_loss���@�1qW(       �pJ	��]��t�A�*

training_accuracy �J?�|%F$       B+�M	f����t�A�*

training_lossWj�@��R(       �pJ	�����t�A�*

training_accuracy�VI?:�F$       B+�M	1���t�A�*

training_loss̿�@l�7(       �pJ	����t�A�*

training_accuracyI2J?����$       B+�M	��3��t�A�*

training_losso�@�SM(       �pJ	��3��t�A�*

training_accuracy%�K?Ē$       B+�M	�Hz��t�A�*

training_lossa͛@��@�(       �pJ	Jz��t�A�*

training_accuracyIK?`��&       sO� 	����t�A�*

validation_loss�ڍ@�e�*       ����	Z����t�A�*

validation_accuracy�6^?�`�$       B+�M	��ڶ�t�A�*

training_loss
S�@��u-(       �pJ	)�ڶ�t�A�*

training_accuracy �J?)���$       B+�M	�F"��t�A�*

training_loss�Ǚ@5e��(       �pJ	�G"��t�A�*

training_accuracy%	L?����$       B+�M	��g��t�A�*

training_loss൛@���(       �pJ	��g��t�A�*

training_accuracy�6J?�8��$       B+�M	�_���t�A�*

training_loss�=�@�Ԉ(       �pJ	a���t�A�*

training_accuracy��K?�cs$       B+�M	���t�A�*

training_loss�0�@��w�(       �pJ	F���t�A�*

training_accuracyn;L?���&       sO� 		v��t�A�*

validation_loss�ˌ@��%(*       ����	Jw��t�A�*

validation_accuracyn�^?r��$       B+�M	�BU��t�A�*

training_loss�C�@F��](       �pJ	�CU��t�A�*

training_accuracy�VJ?���$       B+�M	T���t�A�*

training_loss��@�G(       �pJ	GU���t�A�*

training_accuracyI�K?�ٰ$       B+�M	 ���t�A�*

training_loss ��@��W�(       �pJ	5���t�A�*

training_accuracynK?P`
=$       B+�M	�-��t�A�*

training_losshÙ@CC5�(       �pJ	$-��t�A�*

training_accuracyI�K?j��+$       B+�M	�et��t�A�*

training_loss�O�@�	�(       �pJ	�ft��t�A�*

training_accuracyIRI?��$�&       sO� 	�����t�A�*

validation_loss<��@�I��*       ����	9����t�A�*

validation_accuracy��^?�dp$       B+�M	a7ӹ�t�A�*

training_lossЙ@�¯/(       �pJ	�8ӹ�t�A�*

training_accuracy �J?�g�$       B+�M	R���t�A�*

training_loss���@`6�(       �pJ	����t�A�*

training_accuracyIrK?O�$       B+�M	G`��t�A�*

training_lossa^�@*��(       �pJ	=H`��t�A�*

training_accuracy��I?yn��$       B+�M	�ߦ��t�A�*

training_loss$6�@Q�<U(       �pJ	ᦺ�t�A�*

training_accuracy�MK?Z��m$       B+�M	�\��t�A�*

training_loss��@���(       �pJ	&^��t�A�*

training_accuracyۖJ?���&       sO� 	����t�A�*

validation_loss-��@�"�1*       ����	���t�A�*

validation_accuracy�M_?�=?$       B+�M	��M��t�A�*

training_loss|�@KJ#�(       �pJ	ԄM��t�A�*

training_accuracy�vJ?d޷�$       B+�M	㑖��t�A�*

training_loss��@�_y�(       �pJ	 ����t�A�*

training_accuracy�$J?{�:$       B+�M	�~޻�t�A�*

training_lossu��@r?��(       �pJ	8�޻�t�A�*

training_accuracy%	L?�_��$       B+�M	_1(��t�A�*

training_loss-��@B�(       �pJ	�2(��t�A�*

training_accuracy%iJ?K��$       B+�M	�Lo��t�A�*

training_loss=��@���D(       �pJ	#No��t�A�*

training_accuracy%	K?�_&�&       sO� 	!���t�A�*

validation_loss�:�@���r*       ����	p"���t�A�*

validation_accuracy�V_?���$       B+�M	K�м�t�A�*

training_loss�\�@~c��(       �pJ	��м�t�A�*

training_accuracy�6K?���g$       B+�M	%���t�A�*

training_loss-ڗ@]�v�(       �pJ	]���t�A�*

training_accuracy��K?����$       B+�M	�{^��t�A�*

training_lossq��@�9��(       �pJ	6}^��t�A�*

training_accuracyn;J?�b�x$       B+�M	{w���t�A�*

training_loss�w�@D�e(       �pJ	�x���t�A�*

training_accuracy��K?Di�$       B+�M	�\��t�A�*

training_loss$4�@&��(       �pJ	�]��t�A�*

training_accuracy�L?oΌ:&       sO� 	ޥ��t�A�*

validation_lossB��@�b�u*       ����	"���t�A�*

validation_accuracy�_?����$       B+�M	w�K��t�A�*

training_lossjt�@t��(       �pJ	��K��t�A�*

training_accuracyIrK?cA�$       B+�M	
���t�A�*

training_loss��@^���(       �pJ	E���t�A�*

training_accuracy%�L?g��w$       B+�M	��ھ�t�A�*

training_loss���@�e�(       �pJ	��ھ�t�A�*

training_accuracy  J?ͧk$       B+�M	9z"��t�A�*

training_loss��@����(       �pJ	r{"��t�A�*

training_accuracy�L?a.�x$       B+�M	~k��t�A�*

training_loss��@U 
(       �pJ	ck��t�A�*

training_accuracy��J?f���&       sO� 	Xӂ��t�A�*

validation_loss�@����*       ����	�Ԃ��t�A�*

validation_accuracy  _?Q�q^$       B+�M	��˿�t�A�*

training_loss= �@]�5�(       �pJ	��˿�t�A�*

training_accuracy%�L?�mv�$       B+�M	�*��t�A�*

training_loss?��@c"Y�(       �pJ	7,��t�A�*

training_accuracy%)J?��_$       B+�M	[&Z��t�A�*

training_loss�}�@n���(       �pJ	�'Z��t�A�*

training_accuracy @K?q��g$       B+�M	'Z���t�A�*

training_loss�]�@��e�(       �pJ	^[���t�A�*

training_accuracyI2K?Xߕ�$       B+�M	/���t�A�*

training_lossM.�@�v&3(       �pJ	P0���t�A�*

training_accuracy��K?���&       sO� 	nY��t�A�*

validation_lossP��@��q*       ����	�Z��t�A�*

validation_accuracy  `?��U�$       B+�M	��H��t�A�*

training_loss�"�@l���(       �pJ	��H��t�A�*

training_accuracy �J?���$       B+�M	�����t�A�*

training_lossl�@��-v(       �pJ	1����t�A�*

training_accuracy `K?F���$       B+�M	����t�A�*

training_loss4��@�E�	(       �pJ	����t�A�*

training_accuracy%IK?��$       B+�M	 ��t�A�*

training_loss��@b�r(       �pJ	A ��t�A�*

training_accuracy۶J?`Q�6$       B+�M	JNi��t�A�*

training_loss���@���(       �pJ	�Oi��t�A�*

training_accuracy `K?a��&       sO� 	����t�A�*

validation_loss���@�lV�*       ����	���t�A�*

validation_accuracy%�_?�'$s$       B+�M	����t�A�*

training_loss�@_�w(       �pJ	����t�A�*

training_accuracy%�J?r�O$       B+�M	����t�A�*

training_lossf̕@zg�6(       �pJ	���t�A�*

training_accuracy �K?2�^�$       B+�M	�xW��t�A�*

training_loss6=�@T�q(       �pJ	�yW��t�A�*

training_accuracy �K? �$       B+�M	�F���t�A�*

training_loss�ĕ@���(       �pJ	H���t�A�*

training_accuracyI�L?�t�$       B+�M	���t�A�*

training_lossO�@+^c(       �pJ	Z���t�A�*

training_accuracy��K?,�-c&       sO� 	w:���t�A�*

validation_loss��@����*       ����	�;���t�A�*

validation_accuracyI�^?�3�$       B+�M	�-E��t�A�*

training_loss���@j	~H(       �pJ	�.E��t�A�*

training_accuracy%�K?�$       B+�M	�U���t�A�*

training_loss�F�@�Y��(       �pJ	�V���t�A�*

training_accuracyn{L?���$       B+�M	c4���t�A�*

training_lossʂ�@����(       �pJ	�5���t�A�*

training_accuracyI�K?��N$       B+�M	Sv��t�A�*

training_loss(��@a�
�(       �pJ	�w��t�A�*

training_accuracynL?��5#$       B+�M	7�d��t�A�*

training_loss���@�*��(       �pJ	r�d��t�A�*

training_accuracyIL?��s&       sO� 	�|��t�A�*

validation_lossV�@/t2*       ����	�|��t�A�*

validation_accuracy�_?�o��$       B+�M	�����t�A�*

training_lossTV�@9aZc(       �pJ	����t�A�*

training_accuracy%IK?M��S$       B+�M	����t�A�*

training_lossט�@���B(       �pJ	����t�A�*

training_accuracy%�L?� ҏ$       B+�M	 �R��t�A�*

training_loss4�@ɵ��(       �pJ	Z�R��t�A�*

training_accuracy%�K?b��$       B+�M	�S���t�A�*

training_loss&�@�P3�(       �pJ	ZU���t�A�*

training_accuracy �K?�ρM$       B+�M	�����t�A�*

training_loss�ٕ@o�a�(       �pJ	����t�A�*

training_accuracyn�L?2z�&       sO� 	;����t�A�*

validation_loss;�@="��*       ����	�����t�A�*

validation_accuracy%i_?T�A�$       B+�M	��B��t�A�*

training_loss���@}�h(       �pJ	�B��t�A�*

training_accuracy�mK?u��$       B+�M	*����t�A�*

training_loss���@�D��(       �pJ	�����t�A�*

training_accuracyn�K?{LAF$       B+�M	�����t�A�*

training_lossTP�@5x�(       �pJ	����t�A�*

training_accuracy�L?���$       B+�M	�U��t�A�*

training_loss�l�@^��(       �pJ	cW��t�A�*

training_accuracyn�L?�4$       B+�M	�Jd��t�A�*

training_loss{�@Ix�j(       �pJ	TLd��t�A�*

training_accuracyIK?0T�&       sO� 	��{��t�A�*

validation_loss�\�@�Sς*       ����	�{��t�A�*

validation_accuracyn{_?���$       B+�M	U���t�A�*

training_loss�o�@%��(       �pJ	����t�A�*

training_accuracyIrL?���$       B+�M	D���t�A�*

training_loss���@9�!�(       �pJ	����t�A�*

training_accuracy �L?O�#$       B+�M	\�S��t�A�*

training_loss�6�@�p
(       �pJ	��S��t�A�*

training_accuracy��K?7ǩv$       B+�M	<���t�A�*

training_lossb3�@��(       �pJ	B=���t�A�*

training_accuracy�MM?	��$       B+�M	����t�A�*

training_loss�ē@�5�(       �pJ	R����t�A�*

training_accuracy  M?D��u&       sO� 	%=���t�A�*

validation_loss�M�@���*       ����	�>���t�A�*

validation_accuracyn{_?�w��$       B+�M	Q�C��t�A�*

training_loss*��@�g�(       �pJ	��C��t�A�*

training_accuracy�$M?Hʈc$       B+�M	�;���t�A�*

training_loss4T�@��(       �pJ	n=���t�A�*

training_accuracyn;L?4ݎ,$       B+�M	?����t�A�*

training_loss�"�@F��Y(       �pJ	�����t�A�*

training_accuracyIrK?8�n`$       B+�M	���t�A�*

training_lossg�@��(       �pJ	��t�A�*

training_accuracy��K?�r�$       B+�M	c�^��t�A�*

training_loss*��@[��(       �pJ	��^��t�A�*

training_accuracy�DL?���u&       sO� 	~^x��t�A�*

validation_loss��@�*       ����	�_x��t�A�*

validation_accuracy%�_?�wc�$       B+�M	sl���t�A�*

training_lossY�@�U�(       �pJ	�m���t�A�*

training_accuracyn�L?Zk�$       B+�M	?i��t�A�*

training_loss���@�<��(       �pJ	�j��t�A�*

training_accuracy%iL?���Z$       B+�M	w�N��t�A�*

training_loss&`�@ݼ�(       �pJ	��N��t�A�*

training_accuracy�vJ?�!�$       B+�M	ኖ��t�A�*

training_loss��@x�Ǡ(       �pJ	J����t�A�*

training_accuracyn�M?����$       B+�M	W\���t�A�*

training_loss}є@GYv�(       �pJ	�]���t�A�*

training_accuracyIL?X3n9&       sO� 	j���t�A�*

validation_loss�]�@ݱC*       ����	�k���t�A�*

validation_accuracy��_?��!m$       B+�M	��>��t�A�*

training_loss�0�@����(       �pJ	��>��t�A�*

training_accuracyۖK?��$       B+�M	:n���t�A�*

training_loss���@U�(       �pJ	�o���t�A�*

training_accuracy��L?��$       B+�M	����t�A�*

training_loss�Ӓ@�&[�(       �pJ	c����t�A�*

training_accuracy%�L?@B�$       B+�M	����t�A�*

training_loss댔@�K :(       �pJ	F���t�A�*

training_accuracyn;K?����$       B+�M	�]��t�A�*

training_loss�F�@�#�f(       �pJ	X�]��t�A�*

training_accuracy%�K?\ܱ�&       sO� 	�w��t�A�*

validation_loss�͉@�w�[*       ����	x�w��t�A�*

validation_accuracyn�_?:8�m$       B+�M	�����t�A�*

training_loss`X�@%��(       �pJ	䔿��t�A�*

training_accuracyIRK?�^�$       B+�M	jU��t�A�*

training_loss�'�@(d(       �pJ	�V��t�A�*

training_accuracy�ML?���$       B+�M	��N��t�A�*

training_lossJS�@�΍7(       �pJ	�N��t�A�*

training_accuracyI�J?%q�C$       B+�M	{���t�A�*

training_loss���@�K�(       �pJ	����t�A�*

training_accuracyIK?q���$       B+�M	�v���t�A�*

training_loss���@*�G(       �pJ	9x���t�A�*

training_accuracy �L?��m&       sO� 	�����t�A�*

validation_loss�J�@k��R*       ����	3����t�A�*

validation_accuracy%�_?MW��$       B+�M	F<��t�A�*

training_loss1�@~ ּ(       �pJ	�<��t�A�*

training_accuracy%�L?H*�J$       B+�M	�����t�A�*

training_lossl��@G�Zr(       �pJ	Ǩ���t�A�*

training_accuracy @K?�C�$       B+�M	����t�A�*

training_loss���@�;Y(       �pJ	U����t�A�*

training_accuracy%	L?��$       B+�M	��t�A�*

training_loss���@�!c(       �pJ	R��t�A�*

training_accuracy��K?.p��$       B+�M	s2X��t�A�*

training_loss�+�@�K��(       �pJ	�3X��t�A�*

training_accuracyI�L?0�,&       sO� 	}�q��t�A�*

validation_lossФ�@p�!.*       ����	��q��t�A�*

validation_accuracyI�_?��$       B+�M	c;���t�A�*

training_loss�M�@����(       �pJ	�<���t�A�*

training_accuracyn�L?��t_$       B+�M	�����t�A�*

training_loss��@Rݚ�(       �pJ	�����t�A�*

training_accuracy�L?[�=�$       B+�M	��F��t�A�*

training_loss�:�@"���(       �pJ	)�F��t�A�*

training_accuracy �K?��g�$       B+�M	�ʍ��t�A�*

training_loss��@Qm(       �pJ	�ˍ��t�A�*

training_accuracy  M?7�'�$       B+�M	�����t�A�*

training_lossy��@�Sq(       �pJ	�����t�A�*

training_accuracy @M?7&       sO� 	L6���t�A�*

validation_loss�ʉ@��Xu*       ����	�7���t�A�*

validation_accuracy�M_?�[�Z$       B+�M	�3��t�A�*

training_lossv>�@.@(       �pJ	Q�3��t�A�*

training_accuracy%�L?Z0| $       B+�M	�y��t�A�*

training_loss�@l��(       �pJ	N�y��t�A�*

training_accuracy%�K?��"l$       B+�M	����t�A�*

training_loss�ޓ@�X:(       �pJ	`����t�A�*

training_accuracynL?iLߨ$       B+�M	F#	��t�A�*

training_loss�h�@"��(       �pJ	�$	��t�A�*

training_accuracyn�K?˚�
$       B+�M	�O��t�A�*

training_lossZ4�@dm-�(       �pJ	E�O��t�A�*

training_accuracy%	L?�&�&       sO� 	�g��t�A�*

validation_loss�1�@Zj�;*       ����	)g��t�A�*

validation_accuracy�`?=�%$       B+�M	�����t�A�*

training_loss�z�@�F��(       �pJ	?����t�A�*

training_accuracy�dM?ۥ�$       B+�M	Hd���t�A�*

training_lossTq�@u6��(       �pJ	�e���t�A�*

training_accuracy @M?�6�$       B+�M	%EF��t�A�*

training_loss�-�@l嵡(       �pJ	dFF��t�A�*

training_accuracy�DM?X�/m$       B+�M	����t�A�*

training_loss�I�@�[�(       �pJ	0����t�A�*

training_accuracynM?V��$       B+�M	%����t�A�*

training_losse0�@�U�(       �pJ	h����t�A�*

training_accuracy��L?����&       sO� 	ذ���t�A�*

validation_loss'c�@� �&*       ����	����t�A�*

validation_accuracyۖ_?C!M~$       B+�M	�;=��t�A�*

training_loss�_�@h�2�(       �pJ	�<=��t�A�*

training_accuracyIL?+�8�$       B+�M	x���t�A�*

training_lossK�@�OM(       �pJ	Zy���t�A�*

training_accuracy��L?tIC$       B+�M	nr���t�A�*

training_loss�ۓ@~��(       �pJ	dt���t�A�*

training_accuracyI�K?�"~$       B+�M	����t�A�*

training_loss��@��X(       �pJ	l���t�A�*

training_accuracyIM?�l $       B+�M	2�i��t�A�*

training_loss8��@q��(       �pJ	��i��t�A�*

training_accuracy�mL?.u@t&       sO� 	�4���t�A�*

validation_loss���@�	#/*       ����	6���t�A�*

validation_accuracy�$`?�i.$       B+�M	;����t�A�*

training_loss���@���(       �pJ	�����t�A�*

training_accuracyIRL?E��$       B+�M	KF��t�A�*

training_loss���@���"(       �pJ	�G��t�A�*

training_accuracyn[M?����$       B+�M	��[��t�A�*

training_loss3��@�z�C(       �pJ	>�[��t�A�*

training_accuracyI�L?��.L$       B+�M	[���t�A�*

training_loss�i�@N�^�(       �pJ	����t�A�*

training_accuracyn{L?�Y�j$       B+�M	8:���t�A�*

training_lossTÑ@��*(       �pJ	x;���t�A�*

training_accuracy  M?��;.&       sO� 	���t�A�*

validation_loss@�@
t�r*       ����	)���t�A�*

validation_accuracy @`?ղ��$       B+�M	�H��t�A�*

training_loss?�@�cy$(       �pJ	Q�H��t�A�*

training_accuracy۶K?�(�\$       B+�M	Ά���t�A�*

training_loss�@jԉ(       �pJ	����t�A�*

training_accuracyn�M?���$       B+�M	�"���t�A�*

training_lossy�@j�(       �pJ	2$���t�A�*

training_accuracyIrL?���$       B+�M	�v"��t�A�*

training_loss�M�@�"�(       �pJ	x"��t�A�*

training_accuracy �M?>7�$       B+�M	��l��t�A�*

training_loss��@J��(       �pJ	@�l��t�A�*

training_accuracy �J?�K&       sO� 	|����t�A�*

validation_loss麆@y��*       ����	C����t�A�*

validation_accuracy�V`?�O$       B+�M	�����t�A�*

training_loss!�@����(       �pJ	n����t�A�*

training_accuracy�mL?h}�$       B+�M	�W��t�A�*

training_lossk=�@Ji(       �pJ	Y��t�A�*

training_accuracy۶N?FY>s$       B+�M	�`��t�A�*

training_loss�j�@vڝ(       �pJ	
`��t�A�*

training_accuracy��M?�$       B+�M	 ���t�A�*

training_lossk��@�Az�(       �pJ	����t�A�*

training_accuracy��M?=$$       B+�M	�(���t�A�*

training_loss���@�;�S(       �pJ	�)���t�A�*

training_accuracy��M?S��x&       sO� 	���t�A�*

validation_loss4�@_��*       ����	M���t�A�*

validation_accuracyI2`?ps$       B+�M	�_N��t�A�*

training_loss��@���(       �pJ	@aN��t�A�*

training_accuracy�-O?J�B�$       B+�M	�����t�A�*

training_loss���@�
�(       �pJ	f����t�A�*

training_accuracy `M?��"$       B+�M	x\���t�A�*

training_loss�H�@�E�\(       �pJ	�]���t�A�*

training_accuracy%)L?B_W$       B+�M	�#��t�A�*

training_loss�Ґ@���`(       �pJ	��#��t�A�*

training_accuracy%iL?Y`�$       B+�M	x�k��t�A�*

training_loss��@�+2(       �pJ	��k��t�A�*

training_accuracyn[N?�?W�&       sO� 	ZI���t�A�*

validation_loss�'�@d:�*       ����	�J���t�A�*

validation_accuracy �_?w"�+$       B+�M	*���t�A�*

training_lossn3�@pɱ(       �pJ	����t�A�*

training_accuracy�M?^W��$       B+�M	-H��t�A�*

training_loss*$�@��G0(       �pJ	jI��t�A�*

training_accuracy%iL?d��$       B+�M	��Y��t�A�*

training_lossp��@�[>(       �pJ	��Y��t�A�*

training_accuracyI�K?
�M$       B+�M	,����t�A�*

training_lossF�@���(       �pJ	d����t�A�*

training_accuracy��L?�d[q$       B+�M	`F���t�A�*

training_loss���@	�L�(       �pJ	�G���t�A�*

training_accuracy��L?�h��&       sO� 	���t�A�*

validation_loss:�@�?�*       ����	{���t�A�*

validation_accuracy�-`?� *$       B+�M	��I��t�A�*

training_loss��@����(       �pJ	g�I��t�A�*

training_accuracy��N?L�-c$       B+�M	&����t�A�*

training_loss^�@�5�(       �pJ	�����t�A�*

training_accuracy%�M?�w�$       B+�M	Ci���t�A�*

training_lossr%�@�=�(       �pJ	�j���t�A�*

training_accuracyn�K?r�z�$       B+�M	����t�A�*

training_lossvG�@~�� (       �pJ	&���t�A�*

training_accuracy��L?e��$       B+�M	^�h��t�A�*

training_loss�g�@��LB(       �pJ	��h��t�A�*

training_accuracyn�L?�&       sO� 	ـ��t�A�*

validation_lossͥ�@y��*       ����	�ڀ��t�A�*

validation_accuracy��`?����$       B+�M	'����t�A�*

training_loss���@}��R(       �pJ	�����t�A�*

training_accuracy��L?�k��$       B+�M	Z���t�A�*

training_loss{ϑ@zOB�(       �pJ	����t�A�*

training_accuracy @L?Z���$       B+�M	ÂZ��t�A�*

training_loss���@?X��(       �pJ	(�Z��t�A�*

training_accuracyn�L?�`c$       B+�M	½���t�A�*

training_loss}�@��\�(       �pJ	 ����t�A�*

training_accuracy��K?跴�$       B+�M	i���t�A�*

training_loss�@�%�(       �pJ	����t�A�*

training_accuracy �N?Ϳ��&       sO� 	8U��t�A�*

validation_loss���@�	�"*       ����	�V��t�A�*

validation_accuracyn{`?d�$       B+�M	m�J��t�A�*

training_lossS��@z0E}(       �pJ	��J��t�A�*

training_accuracy%	L?wx�$       B+�M	�u���t�A�*

training_loss�@���7(       �pJ	%w���t�A�*

training_accuracy��M?�ĵS$       B+�M	j����t�A�*

training_loss)
�@�!
�(       �pJ	�����t�A�*

training_accuracy @M?�*�$       B+�M	'� ��t�A�*

training_loss���@sج(       �pJ	a� ��t�A�*

training_accuracyn�M?� $       B+�M	�ah��t�A�*

training_loss�ɏ@;���(       �pJ	1ch��t�A�*

training_accuracyn�L?
���&       sO� 	`����t�A�*

validation_loss^ц@���*       ����	�����t�A�*

validation_accuracyn�`?-�B$       B+�M	>]���t�A�*

training_loss�C�@6Cdf(       �pJ	�^���t�A�*

training_accuracyn;L?Y�$       B+�M	��t�A�*

training_loss3�@�kUn(       �pJ	v	��t�A�*

training_accuracy��M?yj$       B+�M	��]��t�A�*

training_loss�d�@��(       �pJ	S�]��t�A�*

training_accuracyۖM?ֈb�$       B+�M	$~���t�A�*

training_loss`��@��ؕ(       �pJ	}���t�A�*

training_accuracyn;M?5���$       B+�M	�����t�A�*

training_lossQ=�@2~~(       �pJ	�����t�A�*

training_accuracy��O?c`t&       sO� 	c@��t�A�*

validation_loss�q�@2{��*       ����	�A��t�A�*

validation_accuracy�`?���$       B+�M	�P��t�A�*

training_loss�=�@�%(       �pJ	BP��t�A�*

training_accuracyn�M?��л$       B+�M	([���t�A�*

training_loss� �@�^�J(       �pJ	g\���t�A�*

training_accuracy�dN?�NA�$       B+�M	����t�A�*

training_loss�ۍ@��T(       �pJ	L����t�A�*

training_accuracy `M?�	�$       B+�M	'&��t�A�*

training_loss��@�J(       �pJ	g&��t�A�*

training_accuracy��L?���$       B+�M	Φo��t�A�*

training_loss��@�f(       �pJ	�o��t�A�*

training_accuracyIrM?$�v�&       sO� 	x߈��t�A�*

validation_lossm��@�t:=*       ����	�����t�A�*

validation_accuracy��_?��8�$       B+�M	J���t�A�*

training_loss��@��ii(       �pJ	����t�A�*

training_accuracyۖN?w H�$       B+�M	sw��t�A�*

training_loss9�@�G(       �pJ	�x��t�A�*

training_accuracy��M?f|�$       B+�M	w�_��t�A�*

training_lossUݍ@h��v(       �pJ	��_��t�A�*

training_accuracy۶M?�
�_$       B+�M	�e���t�A�*

training_loss֢�@�rY�(       �pJ	�f���t�A�*

training_accuracy��L?�2g$       B+�M	����t�A�*

training_loss�Y�@s��(       �pJ	����t�A�*

training_accuracy��L?���&       sO� 	2���t�A�*

validation_loss�@��5�*       ����	o���t�A�*

validation_accuracyn[`?��q�$       B+�M	�aP��t�A�*

training_loss�@�/L(       �pJ	�bP��t�A�*

training_accuracy�VM?\N9$       B+�M	�����t�A�*

training_loss[�@�QI(       �pJ	ְ���t�A�*

training_accuracyIM?����$       B+�M	<����t�A�*

training_loss$o�@��Ҵ(       �pJ	}����t�A�*

training_accuracy۶M?�X$       B+�M	�)��t�A�*

training_loss�1�@K�y(       �pJ	)��t�A�*

training_accuracyۖM?��"�$       B+�M	gp��t�A�*

training_loss�&�@��:(       �pJ	�p��t�A�*

training_accuracy��M?�v^&       sO� 	�M���t�A�*

validation_loss�ԅ@�D;*       ����	O���t�A�*

validation_accuracy��_?���$       B+�M	�����t�A�*

training_lossԵ�@w�^�(       �pJ	����t�A�*

training_accuracy��M?'`�$       B+�M	���t�A�*

training_loss���@��*b(       �pJ	~���t�A�*

training_accuracy��J?D��$       B+�M	�q_��t�A�*

training_lossDڎ@���0(       �pJ	Ws_��t�A�*

training_accuracyI�L?�837$       B+�M	���t�A�*

training_loss�b�@�⧋(       �pJ	����t�A�*

training_accuracy �N?���$       B+�M	h����t�A�*

training_loss5��@"Eo�(       �pJ	�����t�A�*

training_accuracyI�N?)��&       sO� 	��t�A�*

validation_loss�D�@4�{�*       ����	���t�A�*

validation_accuracy%�_?~?tX$       B+�M	�PN��t�A�*

training_loss��@�|��(       �pJ	jRN��t�A�*

training_accuracy%	O?1,�$       B+�M	��t�A�*

training_loss�ō@%�8(       �pJ	f����t�A�*

training_accuracyn�M?�>�+$       B+�M	5����t�A�*

training_loss`֍@��I(       �pJ	�����t�A�*

training_accuracynM?jU�D$       B+�M	��"��t�A�*

training_loss���@�C�\(       �pJ	ص"��t�A�*

training_accuracy�vL?���$       B+�M	�,j��t�A�*

training_loss���@/�(       �pJ	.j��t�A�*

training_accuracy�DM?��&       sO� 	g{���t�A�*

validation_loss�3�@����*       ����	�|���t�A�*

validation_accuracyۖ`?Da��$       B+�M	����t�A�*

training_loss1��@�: b(       �pJ	����t�A�*

training_accuracyn{K?	�;$       B+�M	���t�A�*

training_loss��@,<�(       �pJ	��t�A�*

training_accuracy%�L?6�R�$       B+�M	�AZ��t�A�*

training_lossz܌@�o��(       �pJ	�BZ��t�A�*

training_accuracy�M? �P�$       B+�M	�ҟ��t�A�*

training_loss��@Nv�(       �pJ	ԟ��t�A�*

training_accuracyn;N?���$       B+�M	�O���t�A�*

training_loss���@�^r (       �pJ	�P���t�A�*

training_accuracy%�M?�2�K&       sO� 	�����t�A�*

validation_loss[�@g�z*       ����	����t�A�*

validation_accuracyn�_?�?��$       B+�M	�F��t�A�*

training_loss@�@� (       �pJ	�F��t�A�*

training_accuracyn�M?���#$       B+�M	a����t�A�*

training_loss72�@> =X(       �pJ	�����t�A�*

training_accuracy%iM?jw�$       B+�M	E����t�A�*

training_loss��@&�q�(       �pJ	�����t�A�*

training_accuracy �M?g^{;$       B+�M	����t�A�*

training_loss�e�@�4I�(       �pJ	Y���t�A�*

training_accuracy��M?	s�$       B+�M	�-a��t�A�*

training_loss��@�~p<(       �pJ	/a��t�A�*

training_accuracy��M?߅<&       sO� 	ly��t�A�*

validation_loss8y�@e��7*       ����	omy��t�A�*

validation_accuracy �_?�#�$       B+�M	?����t�A�*

training_loss��@(Bk�(       �pJ	�����t�A�*

training_accuracy%�M?�k�$       B+�M	�.��t�A�*

training_loss�c�@�SN:(       �pJ	�/��t�A�*

training_accuracyI2N?����$       B+�M	�pP��t�A�*

training_lossz�@�c(       �pJ	8rP��t�A�*

training_accuracy @O?I&.$       B+�M	i=���t�A�*

training_loss�u�@R��(       �pJ	�>���t�A�*

training_accuracynN?���=$       B+�M	�����t�A�*

training_loss�΍@�Y�X(       �pJ	Z����t�A�*

training_accuracyIRM?_Ggg&       sO� 	�W���t�A�*

validation_lossX�@����*       ����	@Y���t�A�*

validation_accuracy �_?�H�N$       B+�M	�.@��t�A�*

training_lossd��@C�[(       �pJ	K0@��t�A�*

training_accuracy @L?�[;$       B+�M	���t�A�*

training_loss.��@T�(       �pJ	Bć��t�A�*

training_accuracy��N?u�|'$       B+�M	r���t�A�*

training_lossϋ@�9��(       �pJ	����t�A�*

training_accuracy%IN?w98$       B+�M	ِ��t�A�*

training_loss�U�@a���(       �pJ	���t�A�*

training_accuracy%�M?n��Z$       B+�M	�]��t�A�*

training_loss���@��c�(       �pJ	��]��t�A�*

training_accuracyn{M?0{#�&       sO� 	8_u��t�A�*

validation_loss�x�@f7�*       ����	w`u��t�A�*

validation_accuracyI�_?�&�$       B+�M	�����t�A�*

training_loss��@�_i(       �pJ	2����t�A�*

training_accuracy��L?g��1$       B+�M	Y��t�A�*

training_loss�Č@TPV�(       �pJ	����t�A�*

training_accuracy%IN?��m@$       B+�M	N��t�A�*

training_losst.�@O>��(       �pJ	^N��t�A�*

training_accuracy�N?h���$       B+�M	�����t�A�*

training_loss�Q�@��Wg(       �pJ	㪕��t�A�*

training_accuracy�VM?�b69$       B+�M	5u���t�A�*

training_loss��@=x8�(       �pJ	jv���t�A�*

training_accuracy��N?s�l&       sO� 	D����t�A�*

validation_loss���@̩��*       ����	�����t�A�*

validation_accuracy�`?�G�$       B+�M	��=��t�A�*

training_loss��@�`&�(       �pJ	��=��t�A�*

training_accuracy��M?�)($       B+�M	٭���t�A�*

training_loss6��@9�(       �pJ	����t�A�*

training_accuracy �N?�4U_$       B+�M	�c���t�A�*

training_loss�	�@�c�(       �pJ	e���t�A�*

training_accuracy `M?к��$       B+�M	/��t�A�*

training_loss���@�J�(       �pJ	���t�A�*

training_accuracyI�M?��$       B+�M	��\��t�A�*

training_lossdǋ@��4E(       �pJ	5�\��t�A�*

training_accuracy�mN?a/7�&       sO� 	��t��t�A�*

validation_loss��@J���*       ����	��t��t�A�*

validation_accuracy�_?`���$       B+�M	�P���t�A�*

training_lossR�@ܾ�(       �pJ	1R���t�A�*

training_accuracyn�N?H)$       B+�M	�Q��t�A�*

training_loss�8�@��y|(       �pJ	&S��t�A�*

training_accuracy%�N?�=�!$       B+�M	��K��t�A�*

training_loss�؊@�P�&(       �pJ	ӠK��t�A�*

training_accuracy%�M?h���$       B+�M	�k���t�A�*

training_loss�D�@�CI(       �pJ	(m���t�A�*

training_accuracy�N?����$       B+�M	�-���t�A�*

training_loss:�@��e_(       �pJ	R/���t�A�*

training_accuracyI�N?�ݰ&       sO� 	�����t�A�*

validation_lossA�@�\~�*       ����	o����t�A�*

validation_accuracy%	`?�t�$       B+�M	7�;��t�A�*

training_lossbȉ@#�7(       �pJ	��;��t�A�*

training_accuracy �N?��#$       B+�M	R+���t�A�*

training_loss1c�@�%:�(       �pJ	�,���t�A�*

training_accuracy%)N?^�I�$       B+�M	#����t�A�*

training_loss*�@��-(       �pJ	�����t�A�*

training_accuracy  M?��sT$       B+�M	.���t�A�*

training_lossg�@����(       �pJ	����t�A�*

training_accuracynM?9�{$       B+�M	J�Y��t�A�*

training_loss��@1�(       �pJ	��Y��t�A�*

training_accuracyn�N?�D�"&       sO� 	r��t�A�*

validation_loss�G�@Y)��*       ����	�	r��t�A�*

validation_accuracy��_?��ר$       B+�M	=���t�A�*

training_losse�@��0(       �pJ	����t�A�*

training_accuracy  M?��$       B+�M	����t�A�*

training_loss݁�@Qk#h(       �pJ	���t�A�*

training_accuracy��M?��
$       B+�M	:�I��t�A�*

training_lossRۊ@��(       �pJ	��I��t�A�*

training_accuracy%�N?���$       B+�M	|p���t�A�*

training_loss*��@��O+(       �pJ	�q���t�A�*

training_accuracy�DN?��$       B+�M	�y���t�A�*

training_loss�)�@��'(       �pJ	e{���t�A�*

training_accuracy%�N?K�4f&       sO� 	+���t�A�*

validation_loss��@l��Y*       ����	x,���t�A�*

validation_accuracy  `?�,d�$       B+�M	�c;��t�A�*

training_lossj�@�(�(       �pJ	�d;��t�A�*

training_accuracy �N?����$       B+�M	�W���t�A�*

training_loss[��@<>�(       �pJ	 Y���t�A�*

training_accuracyn�N?�^�K$       B+�M	�����t�A�*

training_lossT��@���(       �pJ	����t�A�*

training_accuracy��L?Y3�$       B+�M	0���t�A�*

training_loss#ۊ@C=&(       �pJ	����t�A�*

training_accuracyI�M?�C�$       B+�M	;Y��t�A�*

training_lossi�@e�5�(       �pJ	�Y��t�A�*

training_accuracy��N?��&       sO� 	ùp��t�A�*

validation_loss�&�@L��a*       ����	1�p��t�A�*

validation_accuracyn�_?{՜)$       B+�M	n����t�A�*

training_loss#�@5OH(       �pJ	�����t�A�*

training_accuracy �N?d�D$       B+�M	el��t�A�*

training_loss���@�'��(       �pJ	�m��t�A�*

training_accuracy��O?<��$       B+�M	��J��t�A�*

training_loss�%�@�U!�(       �pJ	�J��t�A�*

training_accuracy��O?g���$       B+�M	j~���t�A�*

training_loss�C�@W�(       �pJ	����t�A�*

training_accuracyn{N?B��$       B+�M	����t�A�*

training_loss0��@m��	(       �pJ	'����t�A�*

training_accuracy�DP?H�� &       sO� 	����t�A�*

validation_lossbD�@���*       ����	����t�A�*

validation_accuracy  `?�,�$       B+�M	a>��t�A�*

training_loss�$�@���(       �pJ	�>��t�A�*

training_accuracy%�O?����$       B+�M	����t�A�*

training_loss���@S���(       �pJ	����t�A�*

training_accuracy�mM?�M��$       B+�M	%d���t�A�*

training_loss���@Tʍ�(       �pJ	le���t�A�*

training_accuracy �L?�m�Z$       B+�M	���t�A�*

training_loss�4�@g:(       �pJ	H��t�A�*

training_accuracy�$O?�Sd$       B+�M	�o[��t�A�*

training_loss-��@F�?3(       �pJ	q[��t�A�*

training_accuracy%�O?�㪦&       sO� 	#�s��t�A�*

validation_loss;Ӆ@�O�Y*       ����	b�s��t�A�*

validation_accuracy%I_?��
$       B+�M	�0���t�A�*

training_loss�\�@UD�(       �pJ	(2���t�A�*

training_accuracy�-O?��h$       B+�M	����t�A�*

training_loss�=�@��#�(       �pJ	
���t�A�*

training_accuracy @O?�Ԯ3$       B+�M	1L��t�A�*

training_loss��@�~
|(       �pJ	uL��t�A�*

training_accuracy�MN?h���$       B+�M	�����t�A�*

training_loss�χ@���(       �pJ	ط���t�A�*

training_accuracy%)P?Ɯ�I$       B+�M	�k���t�A�*

training_lossZ��@6U��(       �pJ	�l���t�A�*

training_accuracy�MP?2l �&       sO� 	����t�A�*

validation_loss�Ȇ@�%�;*       ����	�����t�A�*

validation_accuracy۶_?��*E$       B+�M	%; �t�A�*

training_loss�T�@j�V (       �pJ	c; �t�A�*

training_accuracyIO?�CZ$       B+�M	B[� �t�A�*

training_lossg�@�{(       �pJ	\� �t�A�*

training_accuracy�N?�nQs$       B+�M	/� �t�A�*

training_loss�(�@[��)(       �pJ	L0� �t�A�*

training_accuracyn�N?�F$       B+�M	���t�A�*

training_lossEϋ@ٍ�c(       �pJ	��t�A�*

training_accuracy @N? �;$       B+�M	�W�t�A�*

training_loss���@��y�(       �pJ	e�W�t�A�*

training_accuracy�-P?�g�^&       sO� 	��p�t�A�*

validation_loss�!�@��Ϳ*       ����	��p�t�A�*

validation_accuracyI�_?�^�:$       B+�M	ϣ��t�A�*

training_lossʋ�@���(       �pJ	���t�A�*

training_accuracyIO?Q֎�$       B+�M	�0�t�A�*

training_lossiH�@1m'�(       �pJ	'2�t�A�*

training_accuracy�DO?�o��$       B+�M	��I�t�A�*

training_loss���@���6(       �pJ	;�I�t�A�*

training_accuracyn�M?D;�|$       B+�M	����t�A�*

training_lossk�@3��(       �pJ	��t�A�*

training_accuracy�VO?����$       B+�M	����t�A�*

training_loss��@!���(       �pJ	����t�A�*

training_accuracynO?k/n�&       sO� 	hE��t�A�*

validation_loss0u�@F;L�*       ����	�F��t�A�*

validation_accuracy�d_?S���$       B+�M	�S:�t�A�*

training_loss ��@�>��(       �pJ	
U:�t�A�*

training_accuracyIO?�/n�$       B+�M	��t�A�*

training_loss��@E;IR(       �pJ	@��t�A�*

training_accuracy�VN?�I`$       B+�M	���t�A�*

training_loss�(�@u�(       �pJ	&��t�A�*

training_accuracy @P?*��$       B+�M	���t�A�*

training_loss�@�$�(       �pJ	��t�A�*

training_accuracyIrP?�`ke$       B+�M	C�U�t�A�*

training_loss���@}](       �pJ	��U�t�A�*

training_accuracy۶N?e�Ŷ&       sO� 	,/o�t�A�*

validation_loss�Ɇ@��3f*       ����	�0o�t�A�*

validation_accuracy�d_?��;$       B+�M	2���t�A�*

training_loss���@���u(       �pJ	ל��t�A�*

training_accuracyIrO?���\$       B+�M	����t�A�*

training_loss塆@T8wc(       �pJ	���t�A�*

training_accuracyn�O?n/qs$       B+�M	�H�t�A�*

training_lossو@��"&(       �pJ	I
H�t�A�*

training_accuracyI�N?�/�;$       B+�M	4��t�A�*

training_loss�܉@�*�2(       �pJ	<5��t�A�*

training_accuracyI�N?k�q+$       B+�M	�.��t�A�*

training_loss׉@I�(       �pJ	]0��t�A�*

training_accuracyn[N?4c�*&       sO� 	����t�A�*

validation_lossY�@I���*       ����	ϩ��t�A�*

validation_accuracy  _?s�$       B+�M	�`6�t�A�*

training_lossi�@͟B�(       �pJ	Cb6�t�A�*

training_accuracy%	N?��Щ$       B+�M	��~�t�A�*

training_lossh�@��7$(       �pJ	%�~�t�A�*

training_accuracy%	N?���$       B+�M	�5��t�A�*

training_loss�ɇ@��f�(       �pJ	�6��t�A�*

training_accuracy%�N?M��$       B+�M	�%�t�A�*

training_loss�x�@n
-(       �pJ	Q'�t�A�*

training_accuracy��L?̤g$       B+�M	SFT�t�A�*

training_lossa*�@c>*j(       �pJ	�GT�t�A�*

training_accuracyI�P?W��y&       sO� 	��m�t�A�*

validation_loss�x�@��L�*       ����	N�m�t�A�*

validation_accuracy  _?�K��$       B+�M	���t�A�*

training_lossɀ�@�Ĉ�(       �pJ	.��t�A�*

training_accuracynO?eZ�d$       B+�M	��t�A�*

training_loss�S�@��^(       �pJ	� ��t�A�*

training_accuracy%	O?�D�#$       B+�M	�D�t�A�*

training_lossU��@5_��(       �pJ	a�D�t�A�*

training_accuracyn{N?i�.$       B+�M	���t�A�*

training_loss�@Ó6(       �pJ	6��t�A�*

training_accuracy%)O?t�}$       B+�M	��t�A�*

training_loss���@��a�(       �pJ	� ��t�A�*

training_accuracy�6P?R�&       sO� 	X���t�A�*

validation_loss��@yw�*       ����	����t�A�*

validation_accuracy%�^?�Ղb$       B+�M	R�4	�t�A�*

training_loss��@	��(       �pJ	Ä4	�t�A�*

training_accuracy%�O?��l�$       B+�M	ā}	�t�A�*

training_lossЮ�@U�I(       �pJ	1�}	�t�A�*

training_accuracyI�N?b2U�$       B+�M	�~�	�t�A�*

training_loss���@�0�(       �pJ	���	�t�A�*

training_accuracy�$O?�`E�$       B+�M	O�
�t�A�*

training_losso�@p��j(       �pJ	��
�t�A�*

training_accuracy%iO?!��9$       B+�M	G�S
�t�A�*

training_loss���@��H!(       �pJ	��S
�t�A�*

training_accuracy%�P?�Pz&       sO� 	&k
�t�A�*

validation_loss��@��4�*       ����	Sk
�t�A�*

validation_accuracy�6^?q��$       B+�M	BC�
�t�A�*

training_loss
�@_ĉ+(       �pJ	�D�
�t�A�*

training_accuracy�N?�<p$       B+�M	΄�
�t�A�*

training_lossX�@!�u�(       �pJ	8��
�t�A�*

training_accuracy�6O?M�1$       B+�M	�lB�t�A�*

training_lossgG�@�jX�(       �pJ	#nB�t�A�*

training_accuracy%)P?ynĂ$       B+�M	6K��t�A�*

training_loss@��Cr(       �pJ	�L��t�A�*

training_accuracy `O?�'��$       B+�M	d���t�A�*

training_loss!��@Jؿ(       �pJ	Ԭ��t�A�*

training_accuracy�MN?�5��&       sO� 	����t�A�*

validation_lossHB�@i��g*       ����	.���t�A�*

validation_accuracy%�_?`���$       B+�M	_W0�t�A�*

training_loss��@��&n(       �pJ	�X0�t�A�*

training_accuracy �N?"�]�$       B+�M	?�x�t�A�*

training_lossz�@+Y(       �pJ	�x�t�A�*

training_accuracy `N?Y�cJ$       B+�M	�[��t�A�*

training_loss&n�@�	�(       �pJ	]��t�A�*

training_accuracy��O?nE�$       B+�M	I&�t�A�*

training_loss1Ɖ@[��(       �pJ	�'�t�A�*

training_accuracy��N?��}$       B+�M	!�K�t�A�*

training_loss���@�^R�(       �pJ	��K�t�A�*

training_accuracy�6N?6�&       sO� 	(>c�t�A�*

validation_loss@ӆ@�!(*       ����	�?c�t�A�*

validation_accuracy%�_?yB�$       B+�M	�\��t�A�*

training_loss�T�@����(       �pJ	R^��t�A�*

training_accuracy  N?̗�$       B+�M	t��t�A�*

training_loss0�@\}�-(       �pJ	gu��t�A�*

training_accuracy��M?� ��$       B+�M	�8�t�A�*

training_lossDJ�@�-4(       �pJ	u�8�t�A�*

training_accuracy �N?�|�$       B+�M	���t�A�*

training_loss�@�(       �pJ	T��t�A�*

training_accuracy�-O?I��z$       B+�M	����t�A�*

training_lossẍ@�F��(       �pJ	���t�A�*

training_accuracyn[O?WH�&       sO� 	���t�A�*

validation_loss�݅@��b�*       ����	����t�A�*

validation_accuracy��_?�7��$       B+�M	�Z(�t�A�*

training_lossLN�@`t��(       �pJ	\(�t�A�*

training_accuracy��O?��$       B+�M	k�p�t�A�*

training_loss̚�@�~(       �pJ	ؤp�t�A�*

training_accuracyn{P?C��0$       B+�M	���t�A�*

training_lossȄ@,=�v(       �pJ	����t�A�*

training_accuracy�P?AHwo$       B+�M	m���t�A�*

training_lossM؈@��(       �pJ	����t�A�*

training_accuracyn�O?�� D$       B+�M	�E�t�A�*

training_loss���@�7�(       �pJ	[�E�t�A�*

training_accuracy��O?]<�o&       sO� 	 �\�t�A�*

validation_loss�'�@kD�*       ����	��\�t�A�*

validation_accuracy��^?�H�$       B+�M	-j��t�A�*

training_lossr2�@�\�(       �pJ	�k��t�A�*

training_accuracy��O?�=�$       B+�M	���t�A�*

training_loss_��@��&=(       �pJ	����t�A�*

training_accuracyn�N?ss%�$       B+�M	��3�t�A�*

training_lossd��@n��(       �pJ	�3�t�A�*

training_accuracy��N?p\��$       B+�M	�!}�t�A�*

training_loss�B�@�U-(       �pJ	=#}�t�A�*

training_accuracy%�O?���$       B+�M	
���t�A�*

training_loss:0�@�x��(       �pJ	`���t�A�*

training_accuracy%�O?{=&       sO� 	/��t�A�*

validation_loss�͇@6á*       ����	���t�A�*

validation_accuracy��^?1Z�$       B+�M	�=%�t�A�*

training_loss�@�\�(       �pJ	?%�t�A�*

training_accuracyIRP?O�u�$       B+�M	d/l�t�A�*

training_loss�@�3r)(       �pJ	�0l�t�A�*

training_accuracyn{N?O�$       B+�M	�9��t�A�*

training_loss%n�@��P�(       �pJ	�:��t�A�*

training_accuracy�VO?(�M$       B+�M	H��t�A�*

training_losse�@�@��(       �pJ	�I��t�A�*

training_accuracy �P?n�#�$       B+�M	=B�t�A�*

training_loss���@��(       �pJ	�>B�t�A�*

training_accuracy�DN?�:�S&       sO� 	c�Y�t�A�*

validation_loss�n�@�f��*       ����	ѪY�t�A�*

validation_accuracy�6_?�E2~$       B+�M	oy��t�A�*

training_loss�^�@�|�Z(       �pJ	�z��t�A�*

training_accuracy�O?���C$       B+�M	TL��t�A�*

training_loss��@:��(       �pJ	�M��t�A�*

training_accuracy�DN?��g$       B+�M	��1�t�A�*

training_loss��@3ݸ(       �pJ	�1�t�A�*

training_accuracy�-O?���.$       B+�M	��z�t�A�*

training_lossỸ@ùj(       �pJ	H�z�t�A�*

training_accuracyn[P?'Ռ$       B+�M	;���t�A�*

training_loss� �@i���(       �pJ	z���t�A�*

training_accuracy%�O?1-�g&       sO� 	<���t�A�*

validation_lossL(�@E:X*       ����	����t�A�*

validation_accuracy  _?{��n$       B+�M	��#�t�A�*

training_loss�@�@ �(       �pJ	$�#�t�A�*

training_accuracy�dO?�B6$       B+�M	gWk�t�A�*

training_loss�^�@ܒ
�(       �pJ	�Xk�t�A�*

training_accuracy��N?��W$       B+�M	���t�A�*

training_loss�e�@f���(       �pJ	���t�A�*

training_accuracy%�M?��:S$       B+�M	�)��t�A�*

training_lossJ,�@�g��(       �pJ	�*��t�A�*

training_accuracynP?��@$       B+�M	�:D�t�A�*

training_loss?��@Ļ�7(       �pJ	�;D�t�A�*

training_accuracy�dP?��O&       sO� 	B+\�t�A�*

validation_loss�@\�cD*       ����	�,\�t�A�*

validation_accuracyn[^?��$       B+�M	&��t�A�*

training_loss&�@2v)(       �pJ	B'��t�A�*

training_accuracy �O?yY~\$       B+�M	����t�A�*

training_loss�ۆ@YA�T(       �pJ	���t�A�*

training_accuracy �N?��$$       B+�M	zE4�t�A�*

training_loss7�@\e�*(       �pJ	�F4�t�A�*

training_accuracy�vN?�؍N$       B+�M	��}�t�A�*

training_loss$�@~���(       �pJ	R�}�t�A�*

training_accuracy%�N?l5�$       B+�M	�6��t�A�*

training_loss�n�@'\�F(       �pJ	28��t�A�*

training_accuracy��M?��V�&       sO� 	����t�A�*

validation_loss�c�@r���*       ����	���t�A�*

validation_accuracyI�]?)��$       B+�M	ӿ&�t�A�*

training_loss�І@9f��(       �pJ	�&�t�A�*

training_accuracyI�N?rH�$       B+�M	C�n�t�A�*

training_loss{$�@��^�(       �pJ	��n�t�A�*

training_accuracyI2O?F�hU$       B+�M	����t�A�*

training_loss��@lf�2(       �pJ	���t�A�*

training_accuracyn�N?L|t$       B+�M	 �t�A�*

training_loss�/�@v�;�(       �pJ	M �t�A�*

training_accuracy%�P?�W�$       B+�M	/�F�t�A�*

training_lossQm�@�Q�(       �pJ	t�F�t�A�*

training_accuracy��P?�p j&       sO� 	�#^�t�A�*

validation_loss}\�@�v��*       ����	�$^�t�A�*

validation_accuracy `^?~mf�$       B+�M	���t�A�*

training_loss?(�@���n(       �pJ	��t�A�*

training_accuracy��N?�RÍ$       B+�M	���t�A�*

training_loss<Ї@�/R(       �pJ	���t�A�*

training_accuracyI�O?�أH$       B+�M	;�4�t�A�*

training_lossp4�@xj9�(       �pJ	�4�t�A�*

training_accuracyIO?-�o�$       B+�M	��{�t�A�*

training_loss�]�@�t�(       �pJ	�{�t�A�*

training_accuracyIO?���$       B+�M	]C��t�A�*

training_loss�F�@(�\�(       �pJ	�D��t�A�*

training_accuracy�-O?Q��&       sO� 	h���t�A�*

validation_loss�D�@��t�*       ����	����t�A�*

validation_accuracy%	_?�~�$       B+�M	��%�t�A�*

training_loss�_�@uAe(       �pJ	�%�t�A�*

training_accuracy �P?HJ��$       B+�M	�Gm�t�A�*

training_loss<��@Mr�(       �pJ	�Im�t�A�*

training_accuracyI�M?�2S$       B+�M	�|��t�A�*

training_lossT�@���Y(       �pJ	;~��t�A�*

training_accuracy��O?���$       B+�M	c��t�A�*

training_loss`%�@+�!�(       �pJ	���t�A�*

training_accuracyn[O?��#q$       B+�M	MSF�t�A�*

training_lossY>�@
[��(       �pJ	�TF�t�A�*

training_accuracy�P?
�3�&       sO� 	��]�t�A�*

validation_loss��@LDp�*       ����	�]�t�A�*

validation_accuracyn�^?"�m$       B+�M	Q��t�A�*

training_loss�n�@���F(       �pJ	���t�A�*

training_accuracy�MP?,6�$       B+�M	:��t�A�*

training_loss ��@'t�(       �pJ	���t�A�*

training_accuracy��N?8"'�$       B+�M	��1�t�A�*

training_lossjރ@���(       �pJ	�1�t�A�*

training_accuracyn�P?��e$       B+�M	�y�t�A�*

training_loss&}�@�I�(       �pJ	�y�t�A�*

training_accuracy�dO?�l�$       B+�M	����t�A�*

training_loss�݀@5Q�(       �pJ	"���t�A�*

training_accuracy �P?�L�O&       sO� 	L��t�A�*

validation_lossY��@��*%*       ����	|M��t�A�*

validation_accuracyIR^?�K;g$       B+�M	�!�t�A�*

training_loss���@}E�(       �pJ	l�!�t�A�*

training_accuracy�$P?��S$       B+�M	�h�t�A�*

training_loss�z�@n��(       �pJ	��h�t�A�*

training_accuracy%IP?����$       B+�M	Ｏ�t�A�*

training_loss9�@�4�(       �pJ	!���t�A�*

training_accuracy�P?Q%��$       B+�M	.��t�A�*

training_loss��@��B(       �pJ	G/��t�A�*

training_accuracy�DO?��`$       B+�M	�@�t�A�*

training_lossUu�@�jp (       �pJ	(�@�t�A�*

training_accuracy �O?�/a&       sO� 	n�X�t�A�*

validation_loss�G�@nP��*       ����	��X�t�A�*

validation_accuracyn[_?���$       B+�M	�#��t�A�*

training_loss!�@����(       �pJ	�$��t�A�*

training_accuracyn�O?�ܷ�$       B+�M	����t�A�*

training_loss���@���O(       �pJ	����t�A�*

training_accuracy �O?�Ӽ�$       B+�M	��1 �t�A�*

training_loss��@4]�(       �pJ	y�1 �t�A�*

training_accuracy%�O?Șd�$       B+�M		Ex �t�A�*

training_loss���@?���(       �pJ	�Fx �t�A�*

training_accuracy  P?�D�u$       B+�M	Z�� �t�A�*

training_loss�_�@�#E(       �pJ	��� �t�A�*

training_accuracyۖO?Sm,�&       sO� 	.� �t�A�*

validation_loss���@!7s�*       ����	Z� �t�A�*

validation_accuracy۶^?�p�$       B+�M	�!�t�A�*

training_lossѓ�@�<�S(       �pJ	.!�t�A�*

training_accuracy�-P?��U$       B+�M	/g!�t�A�*

training_lossX��@ͷ��(       �pJ	��g!�t�A�*

training_accuracy��N?��/S$       B+�M	�m�!�t�A�*

training_loss�Ʉ@4I��(       �pJ	2o�!�t�A�*

training_accuracy�vP?8d&�$       B+�M	���!�t�A�*

training_loss�Ѓ@C���(       �pJ	A�!�t�A�*

training_accuracy�O?$��5$       B+�M	]!>"�t�A�*

training_loss�߃@�4C�(       �pJ	�">"�t�A�*

training_accuracy�VP?j ��&       sO� 	�U"�t�A�*

validation_loss�d�@Z)�.*       ����	)�U"�t�A�*

validation_accuracy�m^?A-�`$       B+�M	�k�"�t�A�*

training_loss_*�@��i(       �pJ	�l�"�t�A�*

training_accuracy�DP? I�$       B+�M	x�"�t�A�*

training_loss�@,L��(       �pJ	��"�t�A�*

training_accuracy�vO?��B=$       B+�M	��,#�t�A�*

training_loss�J�@5�>�(       �pJ	5�,#�t�A�*

training_accuracy�vP?�M�$       B+�M	-xt#�t�A�*

training_loss9�@�5x((       �pJ	kyt#�t�A�*

training_accuracyn�O?b(�Y$       B+�M	,a�#�t�A�*

training_losse0�@�\�(       �pJ	jb�#�t�A�*

training_accuracyn�P?oP��&       sO� 	�A�#�t�A�*

validation_loss��@�;*       ����	�B�#�t�A�*

validation_accuracyI_?��f$       B+�M	�V$�t�A�*

training_loss
1�@o��X(       �pJ	=X$�t�A�*

training_accuracy�O?}z��$       B+�M	b�e$�t�A�*

training_lossfs�@��(       �pJ	��e$�t�A�*

training_accuracy @P?�k�$       B+�M	6��$�t�A�*

training_loss�@dqr(       �pJ	���$�t�A�*

training_accuracyI�O?�O�$       B+�M	Uv�$�t�A�*

training_loss���@<��(       �pJ	�w�$�t�A�*

training_accuracy�VQ?���r$       B+�M	~q<%�t�A�*

training_loss��@�D�(       �pJ	�r<%�t�A�*

training_accuracyn�O?v�&       sO� 	��S%�t�A�*

validation_loss4O�@1���*       ����	��S%�t�A�*

validation_accuracyn;_?pDA$       B+�M	��%�t�A�*

training_lossc%�@">%�(       �pJ	b��%�t�A�*

training_accuracy%IQ?=���$       B+�M	\��%�t�A�*

training_lossv�@[
�j(       �pJ	־�%�t�A�*

training_accuracy��P?vܪh$       B+�M	�J+&�t�A�*

training_loss��@q�>�(       �pJ	L+&�t�A�*

training_accuracy�6P?�8$       B+�M	7ws&�t�A�*

training_loss���@q�]>(       �pJ	wxs&�t�A�*

training_accuracy��O?ҡ�*$       B+�M	Mۺ&�t�A�*

training_loss�X�@!}lc(       �pJ	�ܺ&�t�A�*

training_accuracy��P?��&       sO� 	��&�t�A�*

validation_loss �@]�y�*       ����	B��&�t�A�*

validation_accuracy��_?H݊$       B+�M	�'�t�A�*

training_lossy?�@�ЮC(       �pJ	G'�t�A�*

training_accuracy `P?�vW�$       B+�M	'4e'�t�A�*

training_loss�Ƃ@�K(       �pJ	�5e'�t�A�*

training_accuracy �O?�|�$       B+�M	c�'�t�A�*

training_loss�M�@���P(       �pJ	���'�t�A�*

training_accuracyIrO?$p�$       B+�M	&��'�t�A�*

training_lossM	�@��(       �pJ	���'�t�A�*

training_accuracy%�N?����$       B+�M	=C=(�t�A�*

training_lossMY�@���(       �pJ	�D=(�t�A�*

training_accuracyn�P?����&       sO� 	��T(�t�A�*

validation_loss0�@��*       ����	&�T(�t�A�*

validation_accuracy�_?M��^$       B+�M	�]�(�t�A�*

training_loss�߁@�K�y(       �pJ	4_�(�t�A�*

training_accuracyI�O?q-a�$       B+�M	�(�t�A�*

training_loss���@���(       �pJ	I�(�t�A�*

training_accuracy%	P?���b$       B+�M	��+)�t�A�*

training_loss�ǁ@�^�(       �pJ	�+)�t�A�*

training_accuracy�mQ?����$       B+�M	`�s)�t�A�*

training_loss%#�@«�l(       �pJ	��s)�t�A�*

training_accuracy�$Q?Sy��$       B+�M	�ǻ)�t�A�*

training_loss���@)I��(       �pJ	ɻ)�t�A�*

training_accuracy�mP?�p2l&       sO� 	-��)�t�A�*

validation_loss���@��*       ����	p��)�t�A�*

validation_accuracy%�_?�Ĉ^$       B+�M	��*�t�A�*

training_loss%�@,��u(       �pJ	��*�t�A�*

training_accuracy�DO?*�}�$       B+�M	x/c*�t�A�*

training_lossq��@ǮE(       �pJ	�0c*�t�A�*

training_accuracy �Q?=��$       B+�M	N��*�t�A�*

training_lossW-�@����(       �pJ	ϸ�*�t�A�*

training_accuracy�DO?�x%�$       B+�M	H�*�t�A�*

training_loss,
�@p�s�(       �pJ	pI�*�t�A�*

training_accuracy��P?���*$       B+�M	fQ<+�t�A�*

training_loss��@k���(       �pJ	�R<+�t�A�*

training_accuracy��Q?o� �&       sO� 	{�S+�t�A�*

validation_loss	�@�֦a*       ����	��S+�t�A�*

validation_accuracy%�_?�7�v$       B+�M	.L�+�t�A�*

training_lossc΂@�p(       �pJ	�M�+�t�A�*

training_accuracyn�O?���.$       B+�M	���+�t�A�*

training_loss�S�@ �<&(       �pJ	]��+�t�A�*

training_accuracy �N?��$       B+�M	2(,�t�A�*

training_loss���@'�g�(       �pJ	�3(,�t�A�*

training_accuracy�$Q?�y��$       B+�M	0�q,�t�A�*

training_lossy��@u
��(       �pJ	{�q,�t�A�*

training_accuracy�O?<�?0$       B+�M	��,�t�A�*

training_loss��@漕V(       �pJ	�,�t�A�*

training_accuracy��P?t��V&       sO� 	=T�,�t�A�*

validation_lossܽ�@B6�U*       ����	�U�,�t�A�*

validation_accuracy%I^?�B�$       B+�M	2l-�t�A�*

training_loss�s�@�U��(       �pJ	�m-�t�A�*

training_accuracyۖO?/��$       B+�M	��_-�t�A�*

training_lossP<�@�a�v(       �pJ	B�_-�t�A�*

training_accuracy�MP?�M�7$       B+�M	�,�-�t�A�*

training_loss��@c��F(       �pJ	b.�-�t�A�*

training_accuracy��P?���$       B+�M	��-�t�A�*

training_loss�>�@44(       �pJ	L�-�t�A�*

training_accuracy��Q?H��5$       B+�M	�8.�t�A�*

training_lossB?�@ag�	(       �pJ	��8.�t�A�*

training_accuracy%	Q?f�b&       sO� 	�MP.�t�A�*

validation_lossȆ@�|�&*       ����	'OP.�t�A�*

validation_accuracy�V^?��$       B+�M	�g�.�t�A�*

training_lossIv�@轎G(       �pJ	]i�.�t�A�*

training_accuracy�MP?I$�$       B+�M	)&�.�t�A�*

training_loss�~@��@�(       �pJ	�'�.�t�A�*

training_accuracy%iR?���$       B+�M	�$/�t�A�*

training_loss�G�@NQ��(       �pJ	@$/�t�A�*

training_accuracy �O?K]NH$       B+�M	�Em/�t�A�*

training_loss�ф@1,��(       �pJ	�Fm/�t�A�*

training_accuracy `O?�]$       B+�M	�6�/�t�A�*

training_loss8W�@�s�(       �pJ	)8�/�t�A�*

training_accuracy%iQ?�*Yy&       sO� 	��/�t�A�*

validation_loss���@ӓ�*       ����	Y��/�t�A�*

validation_accuracy%�^?i�$       B+�M	�0�t�A�*

training_lossJ��@(��(       �pJ	}�0�t�A�*

training_accuracy�P?�R��$       B+�M	S?Z0�t�A�*

training_lossS/�@�o��(       �pJ	�@Z0�t�A�*

training_accuracy �N?i�U�$       B+�M	q��0�t�A�*

training_loss況@N��(       �pJ	��0�t�A�*

training_accuracy��Q?by�$       B+�M	��0�t�A�*

training_lossſ�@	H��(       �pJ	��0�t�A�*

training_accuracy%)P?�B.�$       B+�M	Yq11�t�A�*

training_loss�Z�@ٳS(       �pJ	�r11�t�A�*

training_accuracy�mP?}�Z1&       sO� 	��J1�t�A�*

validation_loss 
�@�
5�*       ����	��J1�t�A�*

validation_accuracyn�]?��}$       B+�M	���1�t�A�*

training_loss,�@�1(       �pJ	���1�t�A�*

training_accuracy�-O?͟�$       B+�M	۬�1�t�A�*

training_loss	��@_�U(       �pJ	F��1�t�A�*

training_accuracy%)P?��t�$       B+�M	#2�t�A�*

training_losso�@��-n(       �pJ	Y$2�t�A�*

training_accuracy��P?�t_�$       B+�M	�i2�t�A�*

training_losse�@���y(       �pJ	i2�t�A�*

training_accuracy�R?@�y$       B+�M	�8�2�t�A�*

training_loss4�@=�!�(       �pJ	�9�2�t�A�*

training_accuracy۶Q?�y��&       sO� 	[&�2�t�A�*

validation_lossw�@�j�*       ����	�'�2�t�A�*

validation_accuracyI�^?�*$       B+�M	�43�t�A�*

training_loss��@�#�(       �pJ	_63�t�A�*

training_accuracyn�P?/΁$       B+�M	k�X3�t�A�*

training_loss���@��%m(       �pJ	��X3�t�A�*

training_accuracy%iP?�d��$       B+�M	���3�t�A�*

training_loss��@$7��(       �pJ	c��3�t�A�*

training_accuracyn[P?�j�$       B+�M	��3�t�A�*

training_loss��~@C�,(       �pJ	+��3�t�A�*

training_accuracy�R?�S�$       B+�M	�-4�t�A�*

training_loss���@�B��(       �pJ	V�-4�t�A�*

training_accuracyn�O?�|*�&       sO� 	��F4�t�A�*

validation_loss��@?`PP*       ����	�F4�t�A�*

validation_accuracy��_?Δ��$       B+�M	�4�t�A�*

training_loss���@���(       �pJ	��4�t�A�*

training_accuracy%	P?�8}r$       B+�M	g��4�t�A�*

training_loss�T~@ذ�(       �pJ	ӈ�4�t�A�*

training_accuracy��Q?��K$       B+�M	Nx5�t�A�*

training_loss8��@�Us�(       �pJ	�y5�t�A�*

training_accuracy @Q?]V(#$       B+�M	r,c5�t�A�*

training_lossV?@u��(       �pJ	�-c5�t�A�*

training_accuracy%	Q?���$       B+�M	�Ԭ5�t�A�*

training_loss�@�پ�(       �pJ	q֬5�t�A�*

training_accuracy��P?Łn�&       sO� 		/�5�t�A�*

validation_lossXA�@\H*       ����	{0�5�t�A�*

validation_accuracy��^?ش��$       B+�M	�O6�t�A�*

training_loss�7�@�N��(       �pJ	�P6�t�A�*

training_accuracy�dQ?��$       B+�M	)WS6�t�A�*

training_lossM.�@� (       �pJ	�XS6�t�A�*

training_accuracy%�Q?����$       B+�M	%��6�t�A�*

training_loss�x}@�1��(       �pJ	���6�t�A�*

training_accuracy�VQ?���$       B+�M	xP�6�t�A�*

training_loss7�{@#K��(       �pJ	�Q�6�t�A�*

training_accuracy�VQ?�o�$$       B+�M	�6+7�t�A�*

training_loss���@b�kk(       �pJ	8+7�t�A�*

training_accuracyn;Q?G��&       sO� 	�\D7�t�A�*

validation_lossiW�@��^�*       ����	<^D7�t�A�*

validation_accuracyۖ]?����$       B+�M	�D�7�t�A�*

training_loss�v@�TB�(       �pJ	F�7�t�A�*

training_accuracy  Q?�׫�$       B+�M	�o�7�t�A�*

training_loss!Ł@0N�(       �pJ	6q�7�t�A�*

training_accuracy%)P?y뒑$       B+�M	Y�8�t�A�*

training_loss�h�@z�(       �pJ	��8�t�A�*

training_accuracyI�O?p��y$       B+�M	��_8�t�A�*

training_loss�ʀ@��2�(       �pJ	� `8�t�A�*

training_accuracyn�Q?���$       B+�M	�ک8�t�A�*

training_loss���@�Cܕ(       �pJ	�۩8�t�A�*

training_accuracyI�P?�Tv&       sO� 		��8�t�A�*

validation_loss;q�@�g]*       ����	S��8�t�A�*

validation_accuracy�M_?�Y��$       B+�M	�9�t�A�*

training_loss7;�@�\�(       �pJ	��9�t�A�*

training_accuracy�MP?X��8$       B+�M	M2R9�t�A�*

training_loss�
�@z�(       �pJ	�3R9�t�A�*

training_accuracy�mP?Ԭ�3$       B+�M	T�9�t�A�*

training_loss2z@�e�(       �pJ	��9�t�A�*

training_accuracy �Q?}mV-$       B+�M	���9�t�A�*

training_loss��@��(       �pJ	���9�t�A�*

training_accuracy%�Q?���$       B+�M	��':�t�A�*

training_loss�)}@���(       �pJ	E�':�t�A�*

training_accuracy `Q?��s�&       sO� 	�@:�t�A�*

validation_loss`r�@���u*       ����	L@:�t�A�*

validation_accuracyn�^?-���$       B+�M	M4�:�t�A�*

training_loss�x{@Ս�3(       �pJ	�5�:�t�A�*

training_accuracyI�Q?�ٟ$       B+�M	��:�t�A�*

training_loss�C�@y�ĭ(       �pJ	s��:�t�A�*

training_accuracy��P?h<��$       B+�M	lc;�t�A�*

training_loss��@��q�(       �pJ	�d;�t�A�*

training_accuracyIRQ?Wr_g$       B+�M	9�^;�t�A�*

training_loss&}@���u(       �pJ	q�^;�t�A�*

training_accuracy�dQ?����$       B+�M	孤;�t�A�*

training_loss$	�@1V�w(       �pJ	��;�t�A�*

training_accuracy �Q?%$]&       sO� 	�;�t�A�*

validation_lossK%�@e�*       ����	L�;�t�A�*

validation_accuracy%i_?���$       B+�M	4;<�t�A�*

training_loss�
~@n�(       �pJ	�<<�t�A�*

training_accuracy `Q?P��$       B+�M	6O<�t�A�*

training_loss�@����(       �pJ	�O<�t�A�*

training_accuracy��P?g���$       B+�M	u��<�t�A�*

training_loss�m�@�,N�(       �pJ	���<�t�A�*

training_accuracy��P?��m$       B+�M	+f�<�t�A�*

training_loss��@"���(       �pJ	�g�<�t�A�*

training_accuracyn;O?3ő$       B+�M	['=�t�A�*

training_loss�~@���(       �pJ	�'=�t�A�*

training_accuracy �Q?��W&       sO� 	'x>=�t�A�*

validation_lossb�@$}D*       ����	ly>=�t�A�*

validation_accuracyn{]?O�%;$       B+�M	�=�t�A�*

training_loss�%~@mzZ�(       �pJ	P�=�t�A�*

training_accuracynQ?w%N$       B+�M	���=�t�A�*

training_loss3�@Q�ȭ(       �pJ	��=�t�A�*

training_accuracy�Q?�ܭ�$       B+�M	c�>�t�A�*

training_loss'̀@�DO(       �pJ	��>�t�A�*

training_accuracy%�P?&��?$       B+�M	�`]>�t�A�*

training_loss�z@�J�j(       �pJ	�a]>�t�A�*

training_accuracy�R?�v�$       B+�M	��>�t�A�*

training_loss��{@����(       �pJ	k��>�t�A�*

training_accuracyI2R?w�~�&       sO� 	�h�>�t�A�*

validation_lossG��@��*       ����	>j�>�t�A�*

validation_accuracy @^?J��$       B+�M	��?�t�A�*

training_loss�F{@2Gm�(       �pJ	��?�t�A�*

training_accuracyn�P?��k�$       B+�M	�L?�t�A�*

training_loss{�@Lt��(       �pJ	@�L?�t�A�*

training_accuracy��P?!($       B+�M	���?�t�A�*

training_loss��@0� (       �pJ	���?�t�A�*

training_accuracyI�Q?��z�$       B+�M	2��?�t�A�*

training_lossO}@91�`(       �pJ	p��?�t�A�*

training_accuracy�dQ?���$       B+�M	y'&@�t�A�*

training_loss�2w@�>j�(       �pJ	�(&@�t�A�*

training_accuracy  S?_�o &       sO� 	4>@�t�A�*

validation_loss�I�@N�o�*       ����	�>@�t�A�*

validation_accuracy��]?��$       B+�M	�؇@�t�A�*

training_loss�\~@�oXa(       �pJ	Dڇ@�t�A�*

training_accuracy۶Q?�9a$       B+�M	�c�@�t�A�*

training_loss��@��]�(       �pJ	Ee�@�t�A�*

training_accuracy��Q?��,}$       B+�M	͜A�t�A�*

training_losskjz@|5�(       �pJ	�A�t�A�*

training_accuracy%)R?W�$       B+�M	�B]A�t�A�*

training_loss��}@��(       �pJ	�C]A�t�A�*

training_accuracyI�P?6q$f$       B+�M	�7�A�t�A�*

training_loss��{@\�p(       �pJ	�8�A�t�A�*

training_accuracyI�P?�4�[&       sO� 	W��A�t�A�*

validation_lossY�@a@�Z*       ����	���A�t�A�*

validation_accuracy��]?�/��$       B+�M	z%B�t�A�*

training_lossX�}@�^��(       �pJ	�&B�t�A�*

training_accuracy @Q?O���$       B+�M	P&MB�t�A�*

training_loss�|@xd3�(       �pJ	�'MB�t�A�*

training_accuracy��P?+8�$       B+�M	�ٖB�t�A�*

training_loss��{@`�:E(       �pJ	EۖB�t�A�*

training_accuracy��R?1��F$       B+�M	���B�t�A�*

training_loss=,~@ثo5(       �pJ	U��B�t�A�*

training_accuracy�VQ?�H��$       B+�M	��'C�t�A�*

training_loss:{@C�DK(       �pJ	��'C�t�A�*

training_accuracy%	R?�#V�&       sO� 	�K?C�t�A�*

validation_lossgD�@CRq:*       ����	5M?C�t�A�*

validation_accuracy�m]?L���$       B+�M	��C�t�A�*

training_loss9Zz@��>�(       �pJ	��C�t�A�*

training_accuracy%IQ?�-_$       B+�M	\��C�t�A�*

training_lossHw@��w�(       �pJ	���C�t�A�*

training_accuracyI2S?�?I$       B+�M	�XD�t�A�*

training_lossaJy@�hs(       �pJ	ZD�t�A�*

training_accuracy�6Q?;��=$       B+�M	�^D�t�A�*

training_loss�@�3�L(       �pJ	b�^D�t�A�*

training_accuracy @P?���[$       B+�M	�D�t�A�*

training_loss�V@�$
�(       �pJ	K��D�t�A�*

training_accuracy `Q?��&       sO� 	jY�D�t�A�*

validation_loss{"�@�'Q*       ����	�Z�D�t�A�*

validation_accuracyI^?~"�&$       B+�M	
E�t�A�*

training_loss��@B��k(       �pJ	RE�t�A�*

training_accuracy �P?n���$       B+�M	�ME�t�A�*

training_loss�`v@vl�(       �pJ	ĀME�t�A�*

training_accuracy��R?7ƹ;$       B+�M	�{�E�t�A�*

training_loss�uw@.J��(       �pJ	}�E�t�A�*

training_accuracy�$R?j��4$       B+�M	$��E�t�A�*

training_loss�W|@^��(       �pJ	f��E�t�A�*

training_accuracy%	R?�)j$       B+�M	4�%F�t�A�*

training_losseNy@u�:F(       �pJ	��%F�t�A�*

training_accuracy��Q?މ�&       sO� 	Z?F�t�A�*

validation_lossc��@#�u*       ����	�[?F�t�A�*

validation_accuracy��]?����$       B+�M	���F�t�A�*

training_loss��z@�}��(       �pJ	R��F�t�A�*

training_accuracy��Q?�D��$       B+�M	�F�t�A�*

training_loss�Zz@�Z�|(       �pJ	Z�F�t�A�*

training_accuracy `Q?)R�/$       B+�M	 +G�t�A�*

training_lossP�|@�d\(       �pJ	G,G�t�A�*

training_accuracy�VR?���$       B+�M	�w^G�t�A�*

training_lossf�@yi��(       �pJ	�x^G�t�A�*

training_accuracy�P?0�/$       B+�M	��G�t�A�*

training_loss{@=��H(       �pJ	��G�t�A�*

training_accuracy۶Q?_�I�&       sO� 	2N�G�t�A�*

validation_losscg�@eN�a*       ����	rO�G�t�A�*

validation_accuracyI�]?*��$       B+�M	eH�t�A�*

training_losshg~@�Wz(       �pJ	ZfH�t�A�*

training_accuracyIrR?0A�$       B+�M	l�NH�t�A�*

training_loss��y@�>��(       �pJ	ĔNH�t�A�*

training_accuracy�R?��$       B+�M	�H�H�t�A�*

training_loss@��Q�(       �pJ	 J�H�t�A�*

training_accuracy��P?�blg$       B+�M	�u�H�t�A�*

training_loss �}@���q(       �pJ	fw�H�t�A�*

training_accuracy%�Q?0�1�$       B+�M	X�'I�t�A�*

training_lossV�}@�V!m(       �pJ	��'I�t�A�*

training_accuracynQ?pP&       sO� 	�"AI�t�A�*

validation_loss���@6��*       ����	2$AI�t�A�*

validation_accuracy�-^?1��|$       B+�M	��I�t�A�*

training_loss{|@��f(       �pJ	��I�t�A�*

training_accuracy��Q?�;E$       B+�M	ʓ�I�t�A�*

training_lossz�y@^�(       �pJ	��I�t�A�*

training_accuracyn;S?�{�$       B+�M	%
J�t�A�*

training_loss�}}@Z�{�(       �pJ	aJ�t�A�*

training_accuracy �Q?��&�$       B+�M	*v^J�t�A�*

training_loss'z@(���(       �pJ	dw^J�t�A�*

training_accuracyn�Q?�sA�$       B+�M	ᔥJ�t�A�*

training_loss�w@ �(       �pJ	+��J�t�A�*

training_accuracy `S?Mkh�&       sO� 	��J�t�A�*

validation_loss���@P�vu*       ����	�J�t�A�*

validation_accuracy%�]?#d
$       B+�M	�K�t�A�*

training_losscE}@�z(       �pJ	)K�t�A�*

training_accuracyIRQ?1��$       B+�M	z3NK�t�A�*

training_loss�)y@ې�!(       �pJ	�4NK�t�A�*

training_accuracy%	Q?�]��$       B+�M	���K�t�A�*

training_lossXGy@,��(       �pJ	1��K�t�A�*

training_accuracyn�R?t��;$       B+�M	z��K�t�A�*

training_lossթx@g�,�(       �pJ	���K�t�A�*

training_accuracy��R?*hVL$       B+�M	��'L�t�A�*

training_loss0I}@1��(       �pJ	Ӆ'L�t�A�*

training_accuracy%�Q?Q⻚&       sO� 	��>L�t�A�*

validation_lossֆ@��ͦ*       ����	��>L�t�A�*

validation_accuracy��]?�&�+$       B+�M	��L�t�A�*

training_loss%�z@dUY�(       �pJ	5�L�t�A�*

training_accuracyI�Q?�f	$       B+�M	Q��L�t�A�*

training_lossBy@�F�(       �pJ	���L�t�A�*

training_accuracy �R?Cz4$       B+�M	�eM�t�A�*

training_loss1y@9��(       �pJ	�fM�t�A�*

training_accuracy��R?�E�$       B+�M	��_M�t�A�*

training_lossy�{@h�a�(       �pJ	Ӕ_M�t�A�*

training_accuracyI�R?��n�$       B+�M	�p�M�t�A�*

training_lossN|@�ZD�(       �pJ	�q�M�t�A�*

training_accuracy�Q?��&       sO� 	7��M�t�A�*

validation_loss���@��Zg*       ����	s��M�t�A�*

validation_accuracy�D]?�$       B+�M	/N�t�A�*

training_loss-�y@\,��(       �pJ	W0N�t�A�*

training_accuracynQ?��t�$       B+�M	�hNN�t�A�*

training_loss�6|@
�<(       �pJ	�iNN�t�A�*

training_accuracyIR?W�I&$       B+�M	�O�N�t�A�*

training_loss�@x@�z�(       �pJ	�P�N�t�A�*

training_accuracy��R?�4&�$       B+�M		��N�t�A�*

training_lossE�x@�6f(       �pJ	h��N�t�A�*

training_accuracy @R?�9�~$       B+�M	�]$O�t�A�*

training_loss�t@�r�(       �pJ	M_$O�t�A�*

training_accuracy�6R?q/�l&       sO� 	]A<O�t�A�*

validation_loss��@�nm;*       ����	�B<O�t�A�*

validation_accuracy�_? ��N$       B+�M	;?�O�t�A�*

training_loss�u@>Q��(       �pJ	�@�O�t�A�*

training_accuracy �R?!��$       B+�M	nt�O�t�A�*

training_loss��t@i;ǅ(       �pJ	�u�O�t�A�*

training_accuracy  T?6��+$       B+�M	biP�t�A�*

training_loss�.w@��>�(       �pJ	�jP�t�A�*

training_accuracyI�R?Uy�G$       B+�M	_^P�t�A�*

training_lossL�v@��N(       �pJ	�^P�t�A�*

training_accuracy��Q?��Eo$       B+�M	�=�P�t�A�*

training_loss�Pw@J�´(       �pJ	?�P�t�A�*

training_accuracy��R?O&       sO� 	��P�t�A�*

validation_losst͊@�0k�*       ����	Z��P�t�A�*

validation_accuracy%�^?!ر$       B+�M	WAQ�t�A�*

training_loss1ex@_E��(       �pJ	�BQ�t�A�*

training_accuracy�mR?�`�c$       B+�M	��MQ�t�A�*

training_losspI{@�=�(       �pJ	ɷMQ�t�A�*

training_accuracy��Q?陿�$       B+�M	w��Q�t�A�*

training_lossg�v@&z�p(       �pJ	���Q�t�A�*

training_accuracyIR?m��4$       B+�M	��Q�t�A�*

training_loss�x@Ob�(       �pJ	��Q�t�A�*

training_accuracyn[R?�'U$       B+�M	{�$R�t�A�*

training_loss�|u@(�	V(       �pJ	Ё$R�t�A�*

training_accuracynS??�&       sO� 	qc<R�t�A�*

validation_loss�ʈ@K���*       ����	�d<R�t�A�*

validation_accuracy�6]?���$       B+�M	�6�R�t�A�*

training_loss��y@_�:�(       �pJ	8�R�t�A�*

training_accuracyn�Q?����$       B+�M	�i�R�t�A�*

training_loss]�x@��x(       �pJ	�j�R�t�A�*

training_accuracy�DS? ��$       B+�M	�/S�t�A�*

training_loss�w@�m(       �pJ	&1S�t�A�*

training_accuracy �R?n�� $       B+�M	%_S�t�A�*

training_lossݿx@p�](       �pJ	;&_S�t�A�*

training_accuracyIRR?o�Z$       B+�M	��S�t�A�*

training_loss��y@��aj(       �pJ		�S�t�A�*

training_accuracy�vQ?[��&       sO� 	�]�S�t�A�*

validation_lossPU�@�<0�*       ����	6_�S�t�A�*

validation_accuracy��]?�ߓ�$       B+�M	}T�t�A�*

training_loss|�w@��͐(       �pJ	I~T�t�A�*

training_accuracy�$Q??9��$       B+�M	5�MT�t�A�*

training_losstQ{@x�޳(       �pJ	s�MT�t�A�*

training_accuracy%iQ?�vM�$       B+�M	԰�T�t�A�*

training_loss��x@� �(       �pJ	��T�t�A�*

training_accuracyn[R?�[�e$       B+�M	���T�t�A�*

training_lossU�v@%�](       �pJ	��T�t�A�*

training_accuracyn�R?� $       B+�M	��%U�t�A�*

training_loss��z@� O�(       �pJ	�%U�t�A�*

training_accuracy �P?�i��&       sO� 	��=U�t�A�*

validation_lossg�@~��3*       ����	�=U�t�A�*

validation_accuracy �]?*D$       B+�M	�j�U�t�A�*

training_lossM�v@�{�(       �pJ	�k�U�t�A�*

training_accuracy  R?���+$       B+�M	�~�U�t�A�*

training_lossҘv@�G�5(       �pJ	��U�t�A�*

training_accuracy%)R?��`�$       B+�M	�YV�t�A�*

training_loss��x@e*Z(       �pJ	'[V�t�A�*

training_accuracy @S?�{�$       B+�M	p#`V�t�A�*

training_loss��w@�Pp)(       �pJ	�$`V�t�A�*

training_accuracyn�R?`~�$       B+�M	�V�t�A�*

training_loss�s@�'�o(       �pJ	K�V�t�A�*

training_accuracyI�S?�U)�&       sO� 	�Q�V�t�A�*

validation_loss_�@@���*       ����	�R�V�t�A�*

validation_accuracy�M^?2���$       B+�M	�W�t�A�*

training_lossy_}@��B(       �pJ	j�W�t�A�*

training_accuracynQ?�T�$       B+�M	OOW�t�A�*

training_lossHw@w��>(       �pJ	�OW�t�A�*

training_accuracy�6S?��$       B+�M	\��W�t�A�*

training_lossG`t@����(       �pJ	���W�t�A�*

training_accuracyIrS?�93�$       B+�M	�d�W�t�A�*

training_loss�y@��gq(       �pJ	f�W�t�A�*

training_accuracy%�Q?��I�$       B+�M	�'X�t�A�*

training_loss��x@n��6(       �pJ	�'X�t�A�*

training_accuracy��Q?6��&       sO� 	{U@X�t�A�*

validation_lossyd�@Om��*       ����	�V@X�t�A�*

validation_accuracyI2^?�AC�$       B+�M	�3�X�t�A�*

training_loss�r@�5��(       �pJ	5�X�t�A�*

training_accuracyn�S?����$       B+�M	ϖ�X�t�A�*

training_loss�w@�H�(       �pJ	��X�t�A�*

training_accuracy�VR?l�($       B+�M	��Y�t�A�*

training_loss��w@��R)(       �pJ	�Y�t�A�*

training_accuracyn�R?wZ9$       B+�M	��aY�t�A�*

training_loss��q@	z�:(       �pJ	W�aY�t�A�*

training_accuracyI�S?	���$       B+�M	��Y�t�A�*

training_loss��u@�E@q(       �pJ	�Y�t�A�*

training_accuracy�6R?}&       sO� 	��Y�t�A�*

validation_lossⱊ@L��*       ����	i��Y�t�A�*

validation_accuracy�]? �r�$       B+�M	�@Z�t�A�*

training_loss�{u@��Ɠ(       �pJ	�AZ�t�A�*

training_accuracyI�S?�g�$       B+�M	iOZ�t�A�*

training_loss_1v@W_9f(       �pJ	AjOZ�t�A�*

training_accuracyI�R?U#�$       B+�M	�c�Z�t�A�*

training_loss��u@��(       �pJ	$e�Z�t�A�*

training_accuracy �S?Rg�$       B+�M	�T�Z�t�A�*

training_loss��v@�E�(       �pJ	V�Z�t�A�*

training_accuracyn�R?����$       B+�M	b&[�t�A�*

training_loss�v@�xE�(       �pJ	��&[�t�A�*

training_accuracy��S?���k&       sO� 	
�?[�t�A�*

validation_loss�֋@]&�*       ����	J�?[�t�A�*

validation_accuracy��]?��:2$       B+�M	-S�[�t�A�*

training_lossQ�x@v ��(       �pJ	cT�[�t�A�*

training_accuracy�R?XF��$       B+�M	���[�t�A�*

training_loss��u@@��<(       �pJ	+��[�t�A�*

training_accuracy%�S?��t�$       B+�M	�j\�t�A�*

training_loss��v@���(       �pJ	
l\�t�A�*

training_accuracy�R?��,�$       B+�M	o�_\�t�A�*

training_losshs@IUq(       �pJ	ƪ_\�t�A�*

training_accuracynS?u��R$       B+�M	�G�\�t�A�*

training_loss�w@���(       �pJ	�H�\�t�A�*

training_accuracy�VR?�a��&       sO� 	�\�t�A�*

validation_lossFd�@l�~�*       ����	h�\�t�A�*

validation_accuracy�D]?�aV�$       B+�M	z0]�t�A�*

training_loss��s@�X(       �pJ	�1]�t�A�*

training_accuracy��R?�rRo$       B+�M	��O]�t�A�*

training_loss'�t@�F+6(       �pJ	��O]�t�A�*

training_accuracy�-R?#��$       B+�M	O��]�t�A�*

training_loss�r@��q(       �pJ	���]�t�A�*

training_accuracyI�R?�/_�$       B+�M	�E�]�t�A�*

training_loss>ps@M=�(       �pJ	�F�]�t�A�*

training_accuracyۖR?�W�$       B+�M	��&^�t�A�*

training_loss84u@)y�(       �pJ	;�&^�t�A�*

training_accuracynR?º�{&       sO� 	ȭ@^�t�A�*

validation_lossdԌ@��*       ����	/�@^�t�A�*

validation_accuracy�d]?gӝ�$       B+�M	���^�t�A�*

training_loss��s@{��O(       �pJ	���^�t�A�*

training_accuracyn�R?YM�$       B+�M	tS�^�t�A�*

training_loss��u@���(       �pJ	�T�^�t�A�*

training_accuracyn�R?�`i�$       B+�M	�_�t�A�*

training_loss��t@m?��(       �pJ	Z�_�t�A�*

training_accuracy�dR?za�$       B+�M	�Ta_�t�A�*

training_loss��t@�,L�(       �pJ	�Ua_�t�A�*

training_accuracy��R?�i�$       B+�M	F�_�t�A�*

training_loss��z@߭�(       �pJ	�_�t�A�*

training_accuracyI2R?�m&       sO� 	�U�_�t�A�*

validation_loss2��@��E*       ����	�V�_�t�A�*

validation_accuracyn[]?ku<�$       B+�M	*�`�t�A�*

training_loss�q@�,!(       �pJ	b�`�t�A�*

training_accuracyI�S?w��$       B+�M	YAO`�t�A�*

training_loss/�r@b�o(       �pJ	�BO`�t�A�*

training_accuracyI�S?��p $       B+�M	��`�t�A�*

training_loss�v@�}�;(       �pJ	��`�t�A�*

training_accuracy  R?ц��$       B+�M	$-�`�t�A�*

training_loss��t@�B��(       �pJ	[.�`�t�A�*

training_accuracyI�R?`�Z�$       B+�M	ڭ&a�t�A�*

training_loss�Tx@K~(       �pJ	1�&a�t�A�*

training_accuracy��Q?ꟸ�&       sO� 	��>a�t�A�*

validation_loss�ʊ@�$|Y*       ����	�>a�t�A�*

validation_accuracy��]?��,�$       B+�M	�i�a�t�A�*

training_loss�p@�q�d(       �pJ	�j�a�t�A�*

training_accuracy�$S?�M�I$       B+�M	���a�t�A�*

training_lossS�v@�H$(       �pJ	3��a�t�A�*

training_accuracy�-R?|&�p$       B+�M	֫b�t�A�*

training_loss>v@�y�(       �pJ	�b�t�A�*

training_accuracy  R?�%$       B+�M	 ab�t�A�*

training_lossf�o@Z(�i(       �pJ	\ab�t�A�*

training_accuracy�6S?��fS$       B+�M	���b�t�A�*

training_loss�_r@���(       �pJ	���b�t�A�*

training_accuracy�6S?���&       sO� 	+��b�t�A�*

validation_lossO��@�@T *       ����	n��b�t�A�*

validation_accuracy�M]?�C�$       B+�M	 �c�t�A�*

training_loss�pt@�w��(       �pJ	A�c�t�A�*

training_accuracy�R?�8�C$       B+�M	��Qc�t�A�*

training_lossz�p@���(       �pJ	I�Qc�t�A�*

training_accuracy�vT?�j�$       B+�M	�c�t�A�*

training_loss�y@4�z�(       �pJ	+��c�t�A�*

training_accuracy  R?��6$       B+�M	L�c�t�A�*

training_loss�
t@�r�s(       �pJ	��c�t�A�*

training_accuracyn�R?2���$       B+�M	��'d�t�A�*

training_loss��q@z9t(       �pJ	2�'d�t�A�*

training_accuracyIRT?4��4&       sO� 	�?d�t�A�*

validation_loss���@L��*       ����	�?d�t�A�*

validation_accuracyI�]?Kz�u$       B+�M	4�d�t�A�*

training_losshs@��$7(       �pJ	j�d�t�A�*

training_accuracy�-R?G
�$       B+�M	���d�t�A�*

training_loss@'s@�P��(       �pJ	!��d�t�A�*

training_accuracyIrS?Ndnz$       B+�M	��e�t�A�*

training_loss��r@>�?�(       �pJ	�e�t�A�*

training_accuracyIrS?B�n�$       B+�M	�2ae�t�A�*

training_loss�cq@�U(       �pJ	44ae�t�A�*

training_accuracy�VT?��$       B+�M	��e�t�A�*

training_loss��s@lV��(       �pJ	�e�t�A�*

training_accuracy%�R?E�~�&       sO� 	���e�t�A�*

validation_loss�L�@�ts*       ����	.��e�t�A�*

validation_accuracyn{]?gz��$       B+�M	�
f�t�A�*

training_loss�]u@x��(       �pJ	J�
f�t�A�*

training_accuracyI2R?j��
$       B+�M	`Rf�t�A�*

training_loss��n@Q�[(       �pJ	WaRf�t�A�*

training_accuracy%)T?"��$       B+�M	�f�t�A�*

training_loss\s@ ��(       �pJ	v�f�t�A�*

training_accuracyn{S?�<�$       B+�M	�9�f�t�A�*

training_loss� q@�=��(       �pJ	.;�f�t�A�*

training_accuracy�S?=�$       B+�M	�y(g�t�A�*

training_loss.zw@3}�(       �pJ	�z(g�t�A�*

training_accuracynS?���U&       sO� 	��?g�t�A�*

validation_loss�@)��5*       ����	�?g�t�A�*

validation_accuracy �\?l���$       B+�M	���g�t�A�*

training_lossX�w@0��(       �pJ	b��g�t�A�*

training_accuracy��Q?��`$$       B+�M	J*�g�t�A�*

training_lossb�p@mE�(       �pJ	�+�g�t�A�*

training_accuracy%�S?�OW$       B+�M	/�h�t�A�*

training_loss�7v@���v(       �pJ	��h�t�A�*

training_accuracy�dS?�R�W$       B+�M	Ej`h�t�A�*

training_loss�r@"�E�(       �pJ	�k`h�t�A�*

training_accuracy �S?�{�$       B+�M	;{�h�t�A�*

training_loss� o@!�b6(       �pJ	�|�h�t�A�*

training_accuracy �R?�	��&       sO� 	
��h�t�A�*

validation_lossJ�@U��-*       ����	v��h�t�A�*

validation_accuracy�M]?'�L�$       B+�M	.�i�t�A�*

training_loss��r@�.~(       �pJ	��i�t�A�*

training_accuracy `S?�8�$       B+�M	<�Ni�t�A�*

training_lossQ�q@.D:(       �pJ	��Ni�t�A�*

training_accuracyn�S?L7s$       B+�M	���i�t�A�*

training_loss��p@O�Q�(       �pJ	 �i�t�A�*

training_accuracy�VS?qf��$       B+�M	T��i�t�A�*

training_loss|t@��+D(       �pJ	���i�t�A�*

training_accuracyI2S?A�m$       B+�M	�f#j�t�A�*

training_loss�pq@�4
(       �pJ	h#j�t�A�*

training_accuracy�vS?+�*&       sO� 	l�:j�t�A�*

validation_loss�7�@=�)*       ����	Ա:j�t�A�*

validation_accuracy��]?�X$       B+�M	��j�t�A�*

training_loss�au@�7�S(       �pJ	y�j�t�A�*

training_accuracy�6T?_�$       B+�M	8��j�t�A�*

training_loss�Hr@���(       �pJ	���j�t�A�*

training_accuracy�T?�v$       B+�M	B�k�t�A�*

training_lossQr@�=�(       �pJ	��k�t�A�*

training_accuracy��R?�0��$       B+�M	o�\k�t�A�*

training_loss��n@�^4�(       �pJ	��\k�t�A�*

training_accuracyn�S?�Q�$       B+�M	K*�k�t�A�*

training_lossas@��ߤ(       �pJ	�+�k�t�A�*

training_accuracyI2R?�!��&       sO� 	݁�k�t�A�*

validation_loss7��@�P *       ����	G��k�t�A�*

validation_accuracyIr]?"2h�$       B+�M	��l�t�A�*

training_loss+hp@*o�(       �pJ	B�l�t�A�*

training_accuracy �S?�G��$       B+�M	P:Jl�t�A�*

training_loss!�r@�"��(       �pJ	�;Jl�t�A�*

training_accuracy��S?��X�$       B+�M	Mړl�t�A�*

training_loss��o@;�e�(       �pJ	�ۓl�t�A�*

training_accuracy @T?Ǯױ$       B+�M	���l�t�A�*

training_losssq@��@(       �pJ	���l�t�A�*

training_accuracy��S?���9$       B+�M	G~!m�t�A�*

training_loss�kl@��#i(       �pJ	�!m�t�A�*

training_accuracy�T?�}n&       sO� 	__:m�t�A�*

validation_lossO1�@�"�*       ����	�`:m�t�A�*

validation_accuracy�V\?�uN$       B+�M	�Ám�t�A�*

training_loss׼o@B�o(       �pJ	lŁm�t�A�*

training_accuracy�MS?����$       B+�M	�F�m�t�A�*

training_loss��n@ʴd(       �pJ	�G�m�t�A�*

training_accuracy%�S?��{4$       B+�M	Z�n�t�A�*

training_loss��n@Ʉ�P(       �pJ	��n�t�A�*

training_accuracy �S?i��$       B+�M	��Yn�t�A�*

training_loss�nk@+��(       �pJ	�Yn�t�A�*

training_accuracy �T?��t$       B+�M	&.�n�t�A�*

training_loss	�q@�жr(       �pJ	e/�n�t�A�*

training_accuracyn[S?𭌵&       sO� 	w#�n�t�A�*

validation_loss�z�@G�'*       ����	�$�n�t�A�*

validation_accuracy۶]?���+