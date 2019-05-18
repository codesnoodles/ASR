from keras.backend.tensorflow_backend import set_session
from keras.callbacks import ModelCheckpoint

from general_function.data_generator import Data
from model.SpeechModel import ModelSpeech

import os
import platform as plat
import tensorflow as tf

DATA_PATH = '/home/wuboyong/corpus/'
MODEL_PATH = 'model_speech'
epoch=2
save_step=1000
batch_size=32


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    # 进行配置，使用95%的GPU
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.95
    # config.gpu_options.allow_growth=True   #不全部占满显存, 按需分配
    set_session(tf.Session(config=config))


    if not os.path.exists(MODEL_PATH):
        os.makedirs(MODEL_PATH)

    ms = ModelSpeech(DATA_PATH)
    data = Data()
    model, _ = ms.CreateModel()
