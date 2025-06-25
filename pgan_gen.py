import os 
import glob
import pickle
import numpy as np

import tensorflow as tf

import logging
tf.get_logger().setLevel(logging.ERROR)

# from tqdm import tqdm 
from pg_gan import PGAN

class GANGenerator():
    def __init__(self, model_dir, 
                 gen_batch_size=64):
        self.model_dir = model_dir 
        assert os.path.isdir(self.model_dir)
        self.gen_batch_size = gen_batch_size
    
    def build_model(self):
        with open(f"{self.model_dir}/config.pickle", "rb") as handle:
            model_config = pickle.load(handle)
        pgan_filters = model_config["filters_scale"]*np.array(model_config["filters"])
        self.latent_dim = model_config["latent_dim"]
        
        self.pgan = PGAN(
            latent_dim=self.latent_dim, 
            channels=model_config["input_shape"][-1], 
            filters=pgan_filters.astype(int), 
            d_steps=1,
            is_3d=len(model_config["input_shape"])>3
        )

        self.pgan.compile(
            d_optimizer=None,
            g_optimizer=None,
        )

        ckpt_path = f"{self.model_dir}/ckpts/"
        trained_ckpts = glob.glob(f"{ckpt_path}/pgan_*_stabilize.ckpt.index")
        trained_depth = np.max([int(fd.split("/")[-1].split("_")[1]) for fd in trained_ckpts])

        prefix = '0_init'
        self.pgan.load_weights(f"{ckpt_path}/pgan_{prefix}.ckpt")

        for n_depth in range(1, trained_depth+1):
            self.pgan.n_depth = n_depth
            prefix = f'{n_depth}_fade_in'
            self.pgan.fade_in_generator()
            self.pgan.fade_in_discriminator()

            self.pgan.load_weights(f"{ckpt_path}/pgan_{prefix}.ckpt")

            prefix = f'{n_depth}_stabilize'
            self.pgan.stabilize_generator()
            self.pgan.stabilize_discriminator()

            self.pgan.load_weights(f"{ckpt_path}/pgan_{prefix}.ckpt")

        print('Restored from ', n_depth, ' step stablize')

    def gen_imgs(self, noises):
        batch_idxes = np.arange(0, len(noises), self.gen_batch_size)
        imgs = []
        for b_i in batch_idxes:
            tmp_imgs = self.pgan.generator(noises[b_i:(b_i+self.gen_batch_size)])
            imgs += tmp_imgs.numpy().tolist()
        imgs = np.squeeze(imgs)
        return imgs 