import tensorflow as tf
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
print("GPUs Available:", tf.config.list_physical_devices('GPU'))