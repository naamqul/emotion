?	?l\)m@?l\)m@!?l\)m@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-?l\)m@?o'??	@1????zk@A a??*??Ik?C4?s$@*	4333?=A2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator?B˺?g@!??????X@)?B˺?g@1??????X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch?L?n??!?c ?v??)?L?n??1?c ?v??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism	????=??!?06??h??)2???A???1??/%?mq?:Preprocessing2F
Iterator::Model??R??!!%??$??)"??3?cf?1??}??W?:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap?ډ???g@!\?d??X@){?V???`?1?\?I?Q?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.moderate"?4.4 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI?S7D@QŊ?{ގW@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?o'??	@?o'??	@!?o'??	@      ??!       "	????zk@????zk@!????zk@*      ??!       2	 a??*?? a??*??! a??*??:	k?C4?s$@k?C4?s$@!k?C4?s$@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q?S7D@yŊ?{ގW@?"?
qgradient_tape/model/separable_conv_1_reduction_right2_stem_1/separable_conv2d/DepthwiseConv2dNativeBackpropFilter#DepthwiseConv2dNativeBackpropFilter@Z??Q??!@Z??Q??0"?
qgradient_tape/model/separable_conv_1_reduction_right1_stem_1/separable_conv2d/DepthwiseConv2dNativeBackpropFilter#DepthwiseConv2dNativeBackpropFilter+????P??!6~
?]Q??0"?
qgradient_tape/model/separable_conv_1_reduction_right3_stem_1/separable_conv2d/DepthwiseConv2dNativeBackpropFilter#DepthwiseConv2dNativeBackpropFilter`??????!N????~??0"?
qgradient_tape/model/separable_conv_2_reduction_right1_stem_1/separable_conv2d/DepthwiseConv2dNativeBackpropFilter#DepthwiseConv2dNativeBackpropFilter揉o?4??!HO?|?K??0"?
qgradient_tape/model/separable_conv_2_reduction_right2_stem_1/separable_conv2d/DepthwiseConv2dNativeBackpropFilter#DepthwiseConv2dNativeBackpropFilter?pL???!??u`????0"?
pgradient_tape/model/separable_conv_1_reduction_left1_stem_1/separable_conv2d/DepthwiseConv2dNativeBackpropFilter#DepthwiseConv2dNativeBackpropFilter???w??!"7T/RA??0"?
pgradient_tape/model/separable_conv_2_reduction_left1_stem_1/separable_conv2d/DepthwiseConv2dNativeBackpropFilter#DepthwiseConv2dNativeBackpropFilter??H??!@??????0"?
qgradient_tape/model/separable_conv_2_reduction_right3_stem_1/separable_conv2d/DepthwiseConv2dNativeBackpropFilter#DepthwiseConv2dNativeBackpropFilter??{=???!?u?D4???0"?
pgradient_tape/model/separable_conv_1_reduction_left4_stem_1/separable_conv2d/DepthwiseConv2dNativeBackpropFilter#DepthwiseConv2dNativeBackpropFilter6Z?#4#??!k??g???0"?
pgradient_tape/model/separable_conv_2_reduction_left4_stem_1/separable_conv2d/DepthwiseConv2dNativeBackpropFilter#DepthwiseConv2dNativeBackpropFilter??)????!DF??#(??0Q      Y@Y(we0????a???nL?X@q??????y2	?IS#L?"?

device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
moderate?4.4 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQ2"GPU(: B 