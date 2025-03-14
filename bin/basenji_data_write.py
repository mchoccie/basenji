#!/usr/bin/env python
# Copyright 2019 Calico LLC

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================
from optparse import OptionParser
import os
import sys

import h5py
import numpy as np
import pdb
import datasets
from tqdm.auto import tqdm
import multiprocessing
from multiprocessing import Pool
from datasets import Dataset
import pysam
import torch.profiler as profiler
from basenji_data import ModelSeq
from basenji.dna_io import dna_1hot, dna_1hot_index
from transformers import AutoTokenizer, AutoModel, TFAutoModel
import torch
#os.environ["CUDA_VISIBLE_DEVICES"] = ""
import tensorflow as tf
# Check if CUDA (GPU) is available


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using PyTorch device: {device}")
FASTA_FILE_OBJ = None  # Global variable for pysam.Fastafile
tf_device = "/CPU:0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

#print(f"Using PyTorch device: {device}")
print(f"Using TensorFlow device: {tf_device}")
"""
basenji_data_write.py

Write TF Records for batches of model sequences.

"""

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:False"

tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNA_bert_6", use_fast=True)
#model = AutoModel.from_pretrained("zhihan1996/DNA_bert_6").to(device)

def process_kmers(tokenized_batches):
    global model, device
    all_embeddings_mean = []

    for tokenized_batch in tqdm(tokenized_batches, desc="Processing batches", unit="batch", position=0, leave=True):
        # Ensure the tokenized batch is on the model's device
        tokenized_batch = {key: val.to(model.device) for key, val in tokenized_batch.items()}

        # Forward pass
        with torch.no_grad():
            outputs = model(**tokenized_batch)
            hidden_states = outputs.last_hidden_state

        # Pooling
        embedding_mean = torch.mean(hidden_states, dim=1)

        # Append results
        all_embeddings_mean.append(embedding_mean.cpu())

    # Concatenate batches
    mean_embeddings = torch.cat(all_embeddings_mean, dim=0)

    return mean_embeddings

def init_worker(fasta_file, shared_model, shared_device):
    """ Initializes global FASTA object in each worker process. """
    global model, device 
    global FASTA_FILE_OBJ
    FASTA_FILE_OBJ = pysam.Fastafile(fasta_file)  # Open the file once per worker
    # ✅ Ensure each worker loads the model independently
    model = shared_model
    device = shared_device
    print(f"Worker initialized with shared model on {device}")

def convert_to_tf(mean_embeddings):
  mean_embeddings_tf = tf.convert_to_tensor(mean_embeddings.numpy())
  return mean_embeddings_tf

def split_into_kmers(sequence, k=6):
  return [sequence[i:i+k] for i in range(0, len(sequence) - k + 1, k)]

def load_model():
    """Load the model in the main process and return it."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model on device: {device}")
    
    model = AutoModel.from_pretrained("zhihan1996/DNA_bert_6").to(device)  # ✅ Load model once
    model.share_memory()  # ✅ Share memory across processes
    return model, device


# Helper function to fetch DNA from FASTA
def fetch_dna(fasta_open, chrom, start, end):
    return fasta_open.fetch(chrom, start, end)

def tokenize_kmers(kmers):
  return tokenizer(kmers, return_tensors="pt", padding="longest", truncation=True)


def process_sequence(si, options, model_seqs, targets):
    ###### THIS IS THE OLD CODE BROOOOOOOOOOOOO
    # """Processes a single sequence in parallel and serializes embeddings for TFRecord."""
    # print(f"Processing sequence index: {si}")  # Debugging log

    # global FASTA_FILE_OBJ  # Use the global FASTA file object

    # msi = options.start_i + si
    # mseq = model_seqs[msi]
    # mseq_start = mseq.start - options.extend_bp
    # mseq_end = mseq.end + options.extend_bp

    # # Fetch DNA sequence
    # seq_dna = fetch_dna(FASTA_FILE_OBJ, mseq.chr, mseq_start, mseq_end)

    # print("This is the length of the DNA sequence: {}".format(str(len(seq_dna))))

    # # Generate k-mers
    # kmers = split_into_kmers(seq_dna, k=6)

    # if len(kmers) == 0:
    #     print(f"Skipping empty sequence {si}")
    #     return []  # Skip empty results

    # # Tokenize k-mers
    # tokenized_kmers = tokenize_kmers(kmers)

    # print("This is the length of the tokenized k-mers: {}".format(str(len(tokenized_kmers))))
    
    # # Generate embeddings
    # mean_embeddings = process_kmers([tokenized_kmers])

    # # Convert embeddings to NumPy
    # mean_embeddings_np = mean_embeddings.numpy()

    # # Retrieve target values for the sequence
    # if options.decimals is not None:
    #     targets_si = targets[si].astype('float32')
    #     targets_si = np.around(targets_si, decimals=options.decimals).astype('float16')
    # else:
    #     targets_si = targets[si]

    # assert np.isinf(targets_si).sum() == 0

    # # Serialize data for TFRecord (storing as bytes)
    # def serialize_example(mean_emb, target_vals):
    #     """Creates a tf.train.Example message ready to be written to a file."""
    #     feature = {
    #         'sequence': feature_bytes(mean_emb),  # Stored as raw bytes
    #         'target': feature_bytes(target_vals)  # Stored as raw bytes
    #     }
    #     return tf.train.Example(features=tf.train.Features(feature=feature)).SerializeToString()

    # # Return serialized TFRecord examples
    # serialized_examples = [serialize_example(mean_emb, targets_si) for mean_emb in mean_embeddings_np]
    # return serialized_examples  # ✅ Now returns serialized TFRecord examples



    # THIS IS THE NEW CODE LETS TEST IT OUT
    global FASTA_FILE_OBJ, model, device
    
    msi = options.start_i + si
    mseq = model_seqs[msi]
    mseq_start = mseq.start - options.extend_bp
    mseq_end = mseq.end + options.extend_bp

    # 1) Fetch DNA
    seq_dna = fetch_dna(FASTA_FILE_OBJ, mseq.chr, mseq_start, mseq_end)
    seq_len = len(seq_dna)
    print("This is the length of the DNA sequence: {}".format(str(len(seq_dna))))
    if seq_len <= 6:
        return []  # skip

    # 2) Generate 6-mers with some stride or block logic
    # e.g. if you want a 1kb block, you can break seq_dna in blocks of size 1000
    block_size = 1000
    kmers_per_block = block_size - 6 + 1  # if sliding every single base
    # We'll store all embeddings in a big list, then group them in blocks.

    all_embeddings = []

    # For example, compute embeddings block by block:
    for block_start in range(0, seq_len, block_size):
        block_end = min(block_start + block_size, seq_len)
        block_dna = seq_dna[block_start:block_end]

        # If block too short, skip or handle specially
        if len(block_dna) < 6:
            continue
        
        # Convert to 6-mers
        block_kmers = []
        for i in range(0, len(block_dna) - 6 + 1, 4):
            #print("This is the start {} and this is the end {}".format(i, i+6))
            block_kmers.append(block_dna[i:i+6])

        # Tokenize
        tokenized = tokenizer(block_kmers, return_tensors="pt", padding="longest", truncation=True)
        tokenized = {k: v.to(model.device) for k,v in tokenized.items()}

        with torch.no_grad():
            outputs = model(**tokenized)
            hidden_states = outputs.last_hidden_state  # [batch_size, seq_len, 768]

        # hidden_states.shape = [num_kmers_in_block, max_kmer_length, 768]
        # but for DNABERT, typically max_kmer_length=6 or so. Usually you might just do
        # mean over the token dimension:
        # e.g. embedding per k-mer = mean over the "sequence length" dimension
        # final shape: [num_kmers_in_block, 768]
        # Then we can block-pool them. 
        # Actually, let's do a mean over dimension 1 => hidden_states.mean(dim=1)
        block_kmer_embeddings = hidden_states.mean(dim=1)  # shape: [N, 768]

        # Now pool across all k-mers in this block
        block_embedding = block_kmer_embeddings.mean(dim=0)  # shape: [768]
        # you could also store each k-mer embedding individually if you prefer
        # but let's just keep one vector per block
        all_embeddings.append(block_embedding.cpu().numpy())

    # Now 'all_embeddings' is ~ (num_blocks, 768).
    if not all_embeddings:
        return []
    all_embeddings_np = np.stack(all_embeddings, axis=0)  # shape: [num_blocks, 768]
    print("This is the shape of the embeddings: {}".format(all_embeddings_np.shape))

    # 3) Retrieve target
    if options.decimals is not None:
        targets_si = targets[si].astype('float32')
        targets_si = np.around(targets_si, decimals=options.decimals).astype('float16')
    else:
        targets_si = targets[si]

    # 4) Convert to TFRecord example
    # We'll store a single TF example for this entire chunk,
    # containing the stacked block embeddings plus the target.
    # Or you could store multiple examples, one per block, but that
    # can become large quickly. Usually it's simpler to keep one
    # example per "sequence".
    
    features_dict = {
        # Flatten the block embeddings
        'sequence': feature_bytes(all_embeddings_np),
        'target': feature_bytes(targets_si)
    }

    example = tf.train.Example(features=tf.train.Features(feature=features_dict))
    return [example.SerializeToString()]  # Return as a list with 1 element


    

################################################################################
# main
################################################################################
def main():
  usage = 'usage: %prog [options] <fasta_file> <seqs_bed_file> <seqs_cov_dir> <tfr_file>'
  parser = OptionParser(usage)
  parser.add_option('-d', dest='decimals',
      default=None, type='int',
      help='Round values to given decimals [Default: %default]')
  parser.add_option('-s', dest='start_i',
      default=0, type='int',
      help='Sequence start index [Default: %default]')
  parser.add_option('-e', dest='end_i',
      default=None, type='int',
      help='Sequence end index [Default: %default]')
  parser.add_option('--te', dest='target_extend',
      default=None, type='int', help='Extend targets vector [Default: %default]')
  parser.add_option('-u', dest='umap_npy',
      help='Unmappable array numpy file')
  parser.add_option('--umap_clip', dest='umap_clip',
      default=1, type='float',
      help='Clip values at unmappable positions to distribution quantiles, eg 0.25. [Default: %default]')
  parser.add_option('--umap_tfr', dest='umap_tfr',
      default=False, action='store_true',
      help='Save umap array into TFRecords [Default: %default]')
  parser.add_option('-x', dest='extend_bp',
      default=0, type='int',
      help='Extend sequences on each side [Default: %default]')
  (options, args) = parser.parse_args()

  if len(args) != 4:
    parser.error('Must provide input arguments.')
  else:
    fasta_file = args[0]
    seqs_bed_file = args[1]
    seqs_cov_dir = args[2]
    tfr_file = args[3]
    

  ################################################################
  # read model sequences
  # print(seqs_bed_file)
  # print(options.end_i)
  # print(options.start_i)
  model_seqs = []
  print(tfr_file)
  for line in open(seqs_bed_file):
    a = line.split()
    model_seqs.append(ModelSeq(a[0],int(a[1]),int(a[2]),None))

  if options.end_i is None:
    options.end_i = len(model_seqs)

  num_seqs = options.end_i - options.start_i
  print("This is the number of sequences {}".format(num_seqs))

  ################################################################
  # determine sequence coverage files

  seqs_cov_files = []
  ti = 0
  seqs_cov_file = '%s/%d.h5' % (seqs_cov_dir, ti)
  while os.path.isfile(seqs_cov_file):
    seqs_cov_files.append(seqs_cov_file)
    ti += 1
    seqs_cov_file = '%s/%d.h5' % (seqs_cov_dir, ti)

  if len(seqs_cov_files) == 0:
    print('Sequence coverage files not found, e.g. %s' % seqs_cov_file, file=sys.stderr)
    exit(1)

  seq_pool_len = h5py.File(seqs_cov_files[0], 'r')['targets'].shape[1]
  num_targets = len(seqs_cov_files)

  ################################################################
  # read targets

  # initialize targets
  targets = np.zeros((num_seqs, seq_pool_len, num_targets), dtype='float16')

  # read each target
  for ti in range(num_targets):
    seqs_cov_open = h5py.File(seqs_cov_files[ti], 'r')
    targets[:,:,ti] = seqs_cov_open['targets'][options.start_i:options.end_i,:]
    seqs_cov_open.close()

  ################################################################
  # modify unmappable

  if options.umap_npy is not None and options.umap_clip < 1:
    unmap_mask = np.load(options.umap_npy)

    for si in range(num_seqs):
      msi = options.start_i + si

      # determine unmappable null value
      seq_target_null = np.percentile(targets[si], q=[100*options.umap_clip], axis=0)[0]

      # set unmappable positions to null
      targets[si,unmap_mask[msi,:],:] = np.minimum(targets[si,unmap_mask[msi,:],:], seq_target_null)

  elif options.umap_npy is not None and options.umap_tfr:
    unmap_mask = np.load(options.umap_npy)

  ################################################################
  # write TFRecords

  # open FASTA
  fasta_open = pysam.Fastafile(fasta_file)
  


    # **Multiprocessing Execution**
  print(multiprocessing.cpu_count())
  model, device = load_model()
  num_workers = min(multiprocessing.cpu_count(), 1)  # Use up to 32 cores
  tf_opts = tf.io.TFRecordOptions(compression_type='ZLIB')

  with tf.io.TFRecordWriter(tfr_file, tf_opts) as writer:
      with multiprocessing.Pool(num_workers, initializer=init_worker, initargs=(fasta_file, model, device)) as pool:
          args = [(si, options, model_seqs, targets) for si in range(num_seqs)]  # Create tuples
          
          # Use `imap()` to ensure results are received in order
          for tf_examples in tqdm(pool.starmap(process_sequence, args), total=num_seqs, desc="Processing Sequences"):
              if not tf_examples:
                print(f"⚠️ Warning: No data returned for sequence!")  # Debugging line
                continue  # Skip empty results

              for example in tf_examples:
                
                writer.write(example)  # Bulk write TFRecords in order

  fasta_open.close()
  print(f"All embeddings serialized to {tfr_file}")

  # tf_opts = tf.io.TFRecordOptions(compression_type='ZLIB')

 
  # with tf.io.TFRecordWriter(tfr_file, tf_opts) as writer:
  #     for si in tqdm(range(num_seqs), desc="Processing sequences"):
  #         msi = options.start_i + si
  #         mseq = model_seqs[msi]
  #         mseq_start = mseq.start - options.extend_bp
  #         mseq_end = mseq.end + options.extend_bp

  #         # Read FASTA sequence
  #         seq_dna = fetch_dna(fasta_open, mseq.chr, mseq_start, mseq_end)
  #         print(f"Processing sequence {si}, splitting into k-mers")

  #         # Generate k-mers
  #         kmers = split_into_kmers(seq_dna, k=6)

  #         batch_size = 100  # Adjust based on memory constraints
  #         all_embeddings_mean = []
  #         all_embeddings_max = []

  #         # Process k-mers in batches
  #         for i in range(0, len(kmers), batch_size):
  #             batch_kmers = kmers[i:i + batch_size]

  #             # Tokenize the batch for DNABERT
  #             tokenized_batch = tokenizer(batch_kmers, return_tensors="tf", padding="longest", truncation=True)
              
  #             # Move input to the correct device
  #             with tf.device(tf_device):
  #                 outputs = model(**tokenized_batch)  # DNABERT forward pass
  #                 hidden_states = outputs.last_hidden_state  # Shape: [batch_size, seq_len, hidden_size]

  #                 # Mean pooling
  #                 embedding_mean = tf.reduce_mean(hidden_states, axis=1)  # Shape: [batch_size, hidden_size]
  #                 all_embeddings_mean.append(embedding_mean)

  #                 # Max pooling
  #                 embedding_max = tf.reduce_max(hidden_states, axis=1)  # Shape: [batch_size, hidden_size]
  #                 all_embeddings_max.append(embedding_max)

  #         # Concatenate all batches
  #         all_embeddings_mean = tf.concat(all_embeddings_mean, axis=0)
  #         all_embeddings_max = tf.concat(all_embeddings_max, axis=0)

  #         print("Final mean-pooled embeddings shape:", all_embeddings_mean.shape)
  #         print("Final max-pooled embeddings shape:", all_embeddings_max.shape)

  #         # Convert embeddings to TFRecord-friendly format (float32 for storage efficiency)
  #         def serialize_array(array):
  #             return tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(array).numpy()]))

  #         # Convert embeddings into byte features for TFRecord
  #         features_dict = {
  #             'sequence_mean': serialize_array(all_embeddings_mean),  # Store mean embeddings
  #             'sequence_max': serialize_array(all_embeddings_max),  # Store max embeddings
  #         }

  #         # Handle target values (ensure correct type and compression)
  #         if options.decimals is not None:
  #             targets_si = targets[si].astype('float32')
  #             targets_si = np.around(targets_si, decimals=options.decimals).astype('float16')
  #         else:
  #             targets_si = targets[si]

  #         assert np.isinf(targets_si).sum() == 0
  #         features_dict['target'] = serialize_array(targets_si)

  #         # Include unmappability mask if needed
  #         if options.umap_tfr:
  #             features_dict['umap'] = serialize_array(unmap_mask[msi, :])

  #         # Serialize and write to TFRecord
  #         example = tf.train.Example(features=tf.train.Features(feature=features_dict))
  #         writer.write(example.SerializeToString())

  # fasta_open.close()
  # print("DONEEE")


def tround(a, decimals):
  """ Truncate to the specified number of decimals. """
  return np.true_divide(np.floor(a * 10**decimals), 10**decimals)

def rround(a, decimals):
  """ Round to the specified number of decimals, randomly sampling
      the last digit according to a bernoulli RV. """
  a_dtype = a.dtype
  a = a.astype('float32')
  dec_probs = (a - tround(a, decimals)) * 10**decimals
  dec_bin = np.random.binomial(n=1, p=dec_probs)
  a_dec = tround(a, decimals) + dec_bin / 10**decimals
  return np.around(a_dec.astype(a_dtype), decimals)

def fetch_dna(fasta_open, chrm, start, end):
  """Fetch DNA when start/end may reach beyond chromosomes."""

  # initialize sequence
  seq_len = end - start
  seq_dna = ''

  # add N's for left over reach
  if start < 0:
    seq_dna = 'N'*(-start)
    start = 0

  # get dna
  seq_dna += fasta_open.fetch(chrm, start, end)

  # add N's for right over reach
  if len(seq_dna) < seq_len:
    seq_dna += 'N'*(seq_len-len(seq_dna))

  return seq_dna


def feature_bytes(values):
  """Convert numpy arrays to bytes features."""
  values = values.flatten().tobytes()
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))


def feature_floats(values):
  """Convert numpy arrays to floats features.
     Requires more space than bytes for float16"""
  values = values.flatten().tolist()
  return tf.train.Feature(float_list=tf.train.FloatList(value=values))


################################################################################
# __main__
################################################################################
if __name__ == '__main__':
  multiprocessing.set_start_method("spawn", force=True) 
  main()
