from src import config, data_utils
import numpy as np
import os

spec = np.random.randn(config.N_MELS, config.N_MELS)
out_path = os.path.join(str(config.REAL_DIR), 'debug_test_spec.npy')

print('Attempting to save to:', out_path)
try:
    data_utils.save_spectrogram(spec, out_path)
    print('save_spectrogram completed')
    print('Exists:', os.path.exists(out_path))
    size = os.path.getsize(out_path) if os.path.exists(out_path) else None
    print('Size (bytes):', size)
except Exception as e:
    print('Exception during save:', type(e), e)
    raise
