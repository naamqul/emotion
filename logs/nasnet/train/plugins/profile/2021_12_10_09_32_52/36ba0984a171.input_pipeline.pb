	?l\)m@?l\)m@!?l\)m@      ??!       "n
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
	?o'??	@?o'??	@!?o'??	@      ??!       "	????zk@????zk@!????zk@*      ??!       2	 a??*?? a??*??! a??*??:	k?C4?s$@k?C4?s$@!k?C4?s$@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q?S7D@yŊ?{ގW@