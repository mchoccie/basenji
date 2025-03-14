import tensorflow as tf
import numpy as np

def _parse_function(example_proto):
    """Parses a single TFRecord example."""
    feature_description = {
        'sequence': tf.io.FixedLenFeature([], tf.string),
        'target': tf.io.FixedLenFeature([], tf.string),
    }
    
    parsed_example = tf.io.parse_single_example(example_proto, feature_description)

    # Decode bytes inside TensorFlow Graph Mode
    sequence_array = tf.io.decode_raw(parsed_example['sequence'], tf.float32)
    target_array = tf.io.decode_raw(parsed_example['target'], tf.float16)

    return sequence_array, target_array

def read_tfrecord(file_path):
    """Reads a compressed TFRecord file and decodes its contents."""
    tf_opts = tf.io.TFRecordOptions(compression_type="ZLIB")  # Specify ZLIB decompression
    raw_dataset = tf.data.TFRecordDataset(file_path, compression_type="ZLIB")  # Decompress while reading
    parsed_dataset = raw_dataset.map(_parse_function)

    for seq, target in parsed_dataset:
        print(seq.shape)
        # seq_numpy = seq.numpy().reshape(-1, 201984)  # Adjust based on actual shape
        # target_numpy = target.numpy()

        # print("Sequence Embeddings Shape:", seq_numpy.shape)
        # print("Target Shape:", target_numpy.shape)
        # print("First Sequence Embedding:", seq_numpy[0])  # Print first vector
        # print("First Target Value:", target_numpy[0])  # Print first target value
        # print("-" * 50)

# Example usage
tfrecord_file = "/home/017448899/basenji/manuscripts/akita/data/1m/tfrecords/train-5.tfr"
read_tfrecord(tfrecord_file)