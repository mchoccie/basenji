import tensorflow as tf
print("TensorFlow version:", tf.__version__)
print(tf.sysconfig.get_build_info())
print("cuDNN version:", tf.sysconfig.get_build_info().get("cudnn_version"))
print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))
print("GPU Devices:", tf.config.list_physical_devices('GPU'))
print(tf.test.is_built_with_cuda())