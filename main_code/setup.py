import tensorflow as tf
import os

def setup_gpu():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    print('gpus: ', gpus)
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
        print('cur gpu: ', gpu)

def setup_paths():
    POS_PATH = os.path.join('../data', 'positive')
    NEG_PATH = os.path.join('../data', 'negative')
    ANC_PATH = os.path.join('../data', 'anchor')
    return POS_PATH, NEG_PATH, ANC_PATH
