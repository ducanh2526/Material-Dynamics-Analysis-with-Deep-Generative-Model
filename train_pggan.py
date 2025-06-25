import os
import yaml
import shutil
import glob
import argparse
import pickle
import numpy as np
from math import ceil

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import Callback
from tensorflow.keras import backend 
from keras import backend as K

config = tf.compat.v1.ConfigProto() 
config.gpu_options.allow_growth = True  # don't pre-allocate memory; allocate as-needed
config.gpu_options.per_process_gpu_memory_fraction = 0.3  # limit memory to be allocated
K.set_session(tf.compat.v1.Session(config=config)) # create sess w/ above settings

from pg_gan import PGAN, WeightedSum
from utils.gan_dat_gen import DataIterator
from utils.plot import plot_fn_dict

class GANMonitor(Callback):
    def __init__(self, latent_dim, num_samples=16, plot_fn=None, prefix='', 
                 save_ckpt_dir="../", norm_term=None):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_samples = num_samples
        # self.output_dir = output_dir
        self.plot_fn = plot_fn
        self.prefix = prefix
        self.norm_term = norm_term
        self.random_latent_vectors = tf.random.normal(shape=(num_samples, latent_dim))

        self.save_ckpt_dir = save_ckpt_dir 
        if not os.path.isdir(self.save_ckpt_dir):
            os.makedirs(self.save_ckpt_dir)
        self.ckpt_path = f"{self.save_ckpt_dir}/ckpts/"
        if not os.path.isdir(self.ckpt_path):
            os.makedirs(self.ckpt_path)

    def set_prefix(self, prefix=''):
        self.prefix = prefix

    def set_steps(self, steps_per_epoch, epochs):
        self.steps_per_epoch = steps_per_epoch
        self.epochs = epochs
        self.steps = self.steps_per_epoch * self.epochs

    def on_epoch_begin(self, epoch, logs=None):
        self.n_epoch = epoch

    def on_epoch_end(self, epoch, logs=None):
        if self.plot_fn is not None:
            save_dir = f"{self.save_ckpt_dir}/Visualize/"
            self.plot_fn(
                generator=self.model.generator,
                latent_vectors=self.random_latent_vectors,
                save_path=save_dir,
                epoch=epoch,
                prefix=self.prefix,
                norm_term=self.norm_term  # will be ignored if not needed
            )

    def on_batch_begin(self, batch, logs=None):
        alpha = ((self.n_epoch * self.steps_per_epoch) + batch) / \
            float(self.steps - 1)
        for layer in self.model.generator.layers:
            if isinstance(layer, WeightedSum):
                backend.set_value(layer.alpha, alpha)
        for layer in self.model.discriminator.layers:
            if isinstance(layer, WeightedSum):
                backend.set_value(layer.alpha, alpha)


# def load_config(config_path):
#     with open(config_path, "rb") as f:
#         return pickle.load(f)

def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def train_stage(pgan, cbk, data_path, batch_size, resolution, epochs, prefix, is_fade_in):
    cbk.set_prefix(prefix=prefix)
    cbk.set_steps(steps_per_epoch=ceil(NUM_IMGS / batch_size), epochs=epochs)

    if is_fade_in:
        pgan.fade_in_generator()
        pgan.fade_in_discriminator()
    else:
        pgan.stabilize_generator()
        pgan.stabilize_discriminator()

    pgan.compile(d_optimizer=discriminator_optimizer, g_optimizer=generator_optimizer)

    train_iter = DataIterator(
        data_path=data_path,
        iter_type="train",
        resize_dim=3*(resolution, ) if pgan.is_3d else 2*(resolution, ),
        batch_size=batch_size,
        expand_dims=False
    )

    pgan.fit(x=train_iter, steps_per_epoch=ceil(NUM_IMGS / batch_size),
             epochs=epochs, callbacks=[cbk])

    pgan.save_weights(f"{cbk.ckpt_path}/pgan_{prefix}.ckpt")


def main(config_path):
    global NUM_IMGS, discriminator_optimizer, generator_optimizer

    MODEL_CONFIG = load_config(config_path)
    DATA_PATH = MODEL_CONFIG["data_path"]
    DATA_NAME = DATA_PATH.split("/")[-2]
    NOISE_DIM = MODEL_CONFIG["latent_dim"]
    BATCH_SIZE = MODEL_CONFIG["batch_size"]
    EPOCHS = MODEL_CONFIG["n_epochs"]
    FILTERS = MODEL_CONFIG["filters"]
    INPUT_SHAPE = MODEL_CONFIG["input_shape"]
    plot_fn = plot_fn_dict[MODEL_CONFIG["plot_fn"]]

    NUM_IMGS = len([fn for fn in os.listdir(DATA_PATH) if "npy" in fn])

    generator_optimizer = keras.optimizers.Adam(1e-3, beta_1=0.0, beta_2=0.99, epsilon=1e-8)
    discriminator_optimizer = keras.optimizers.Adam(1e-3, beta_1=0.0, beta_2=0.99, epsilon=1e-8)

    model_name = "{}_nDims{}_epoch{}_mulFil{}_dExtra{}".format(
        MODEL_CONFIG["name"],
        NOISE_DIM,
        EPOCHS,
        MODEL_CONFIG["filters_scale"],
        MODEL_CONFIG["d_extra_steps"]
    )
    print(model_name)

    if os.path.isfile(f"{DATA_PATH}/norm_term.npy"):
        norm_term = np.load(f"{DATA_PATH}/norm_term.npy", allow_pickle=True)
    else:
        norm_term = None

    cbk = GANMonitor(num_samples=16, latent_dim=NOISE_DIM, prefix='0_init',
                     plot_fn=plot_fn,
                     save_ckpt_dir=f"./test/{DATA_NAME}/{model_name}/",
                     norm_term=norm_term)
    
    steps_per_epoch = ceil(NUM_IMGS / BATCH_SIZE[0])
    cbk.set_steps(steps_per_epoch=steps_per_epoch, epochs=EPOCHS)

    os.makedirs(cbk.save_ckpt_dir, exist_ok=True)
    # shutil.copy("train_pggan.py", f"{cbk.save_ckpt_dir}/train_pgan.py")
    # shutil.copy("pg_gan.py", f"{cbk.save_ckpt_dir}/pg_cgan.py")
    with open(f"{cbk.save_ckpt_dir}/config.pickle", "wb") as handle:
        pickle.dump(MODEL_CONFIG, handle, protocol=pickle.HIGHEST_PROTOCOL)

    pgan_filters = MODEL_CONFIG["filters_scale"] * np.array(FILTERS)
    pgan = PGAN(
        latent_dim=NOISE_DIM,
        filters=pgan_filters.astype(int),
        channels=INPUT_SHAPE[-1],
        d_steps=MODEL_CONFIG["d_extra_steps"],
        is_3d=len(INPUT_SHAPE)>3
    )

    pgan.compile(d_optimizer=discriminator_optimizer, g_optimizer=generator_optimizer)

    trained_depth = 0
    if os.path.isdir(cbk.ckpt_path):
        trained_ckpts = glob.glob(f"{cbk.ckpt_path}/pgan_*_stabilize.ckpt.index")
        if len(trained_ckpts) > 0:
            trained_depth = np.max([int(fd.split("/")[-1].split("_")[1]) for fd in trained_ckpts])
            for n_depth in range(1, trained_depth + 1):
                pgan.n_depth = n_depth
                pgan.fade_in_generator()
                pgan.fade_in_discriminator()
                pgan.load_weights(f"{cbk.ckpt_path}/pgan_{n_depth}_fade_in.ckpt")

                pgan.stabilize_generator()
                pgan.stabilize_discriminator()
                pgan.load_weights(f"{cbk.ckpt_path}/pgan_{n_depth}_stabilize.ckpt")

    if trained_depth == 0:
        # Train the initial 4x4 stage
        train_iter = DataIterator(data_path=DATA_PATH, iter_type="train",
                                  expand_dims=False,
                                  resize_dim=(4, 4, 4) if pgan.is_3d else (4, 4),
                                  batch_size=BATCH_SIZE[0])
        pgan.fit(x=train_iter, steps_per_epoch=steps_per_epoch,
                 epochs=EPOCHS, callbacks=[cbk])
        pgan.save_weights(f"{cbk.ckpt_path}/pgan_{cbk.prefix}.ckpt")

    max_depth = int(np.log2(INPUT_SHAPE[0]))
    for n_depth in range(trained_depth + 1, max_depth - 1):
        pgan.n_depth = n_depth
        resolution = 4 * (2 ** n_depth)
        bs = BATCH_SIZE[n_depth]
        epochs = int(EPOCHS * (BATCH_SIZE[0] / bs) / 2)

        train_stage(pgan, cbk, DATA_PATH, bs, resolution, epochs,
                    f"{n_depth}_fade_in", is_fade_in=True)
        train_stage(pgan, cbk, DATA_PATH, bs, resolution, epochs,
                    f"{n_depth}_stabilize", is_fade_in=False)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args.config)
