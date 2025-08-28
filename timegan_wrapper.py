
# timegan_wrapper.py
# This script wraps TimeGAN (from https://github.com/jsyoon0823/TimeGAN) for use in simulations

import os
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from timegan import timegan  # You must place this wrapper in the root of the original TimeGAN repo
from utils import normalizer, train_test_split
from metrics.discriminative import discriminative_score_metrics
from metrics.predictive import predictive_score_metrics

def timegan_main_custom(data, seq_len=24, gen_ratio=1.0, model='gru', epochs=500):
    """
    data: np.ndarray of shape (N, 1), e.g. speed time series
    seq_len: Length of time sequences for TimeGAN
    gen_ratio: How much additional data to generate (1.0 = same size as input)
    model: RNN cell type ('gru', 'lstm', etc.)
    epochs: Number of training epochs
    Returns: generated_data (flattened array)
    """

    # Ensure shape = [# sequences, seq_len, dim]
    total_len = len(data)
    pad_len = seq_len - (total_len % seq_len)
    if pad_len > 0:
        data = np.pad(data, ((0, pad_len)), mode='edge')
    reshaped_data = data.reshape(-1, seq_len, 1)

    # Normalize
    norm_data, norm_params = normalizer(reshaped_data, 'minmax')

    # Run TimeGAN
    parameters = {
        'module': model,
        'hidden_dim': 24,
        'num_layer': 3,
        'iterations': epochs,
        'batch_size': 128,
        'seq_len': seq_len,
        'z_dim': 32
    }

    tf.reset_default_graph()
    generated_data = timegan(norm_data, parameters)

    # Inverse transform
    from utils import renormalization
    generated_data = renormalization(generated_data, norm_params)

    # Flatten output
    flat = generated_data.reshape(-1, generated_data.shape[-1])
    n_generated = int(len(flat) * gen_ratio)
    return flat[:n_generated]
