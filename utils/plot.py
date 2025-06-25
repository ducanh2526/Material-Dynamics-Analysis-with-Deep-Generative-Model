import os
import matplotlib.pyplot as plt
import numpy as np
from math import sqrt

def plot_cxdi_chart(generator, latent_vectors, save_path, epoch, prefix='0_init', norm_term=None):
    samples = generator(latent_vectors, training=False)
    n_grid = int(sqrt(len(latent_vectors)))

    fig, axes = plt.subplots(n_grid, n_grid, figsize=(4*n_grid, 4*n_grid))
    for i in range(n_grid):
        for j in range(n_grid):
            img = samples[i*n_grid+j].numpy()[:, :, 0]
            axes[i][j].imshow(img, cmap='gray', vmin=-1, vmax=1)
            axes[i][j].axis('off')
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(f'{save_path}/plot_chart_{prefix}_{epoch:05d}.png', bbox_inches='tight')
    plt.close(fig)

def plot_cxdi_NP(generator, latent_vectors, save_path, epoch, prefix='0_init', norm_term=None):
    samples = generator(latent_vectors, training=False)
    n_grid = int(sqrt(len(latent_vectors)))

    fig, axes = plt.subplots(n_grid, n_grid, figsize=(4*n_grid, 4*n_grid))
    for i in range(n_grid):
        for j in range(n_grid):
            img = samples[i*n_grid+j].numpy()
            particle = img[:, :, 0].copy()
            background = img[:, :, 1].copy()
            particle[particle > 0] = 0
            axes[i][j].imshow(particle + background, cmap='Greys_r', vmin=-0.5, vmax=0.5)
            axes[i][j].axis('off')
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(f'{save_path}/plot_ptycho_{prefix}_{epoch:05d}.png', bbox_inches='tight')
    plt.close(fig)

def plot_xafs_3d(generator, latent_vectors, save_path, epoch, prefix='0_init', norm_term=None):
    samples = generator(latent_vectors, training=False).numpy()
    if norm_term is not None:
        samples = samples * norm_term + norm_term
    sample_dim = samples.shape[1]
    n_grid = int(sqrt(len(latent_vectors)))
    
    fig = plt.figure(figsize=(4*n_grid, 4*n_grid))
    for i in range(n_grid):
        for j in range(n_grid):
            ax = fig.add_subplot(n_grid, n_grid, i*n_grid+j+1, projection='3d')
            tmp_sample = samples[i*n_grid+j]
            tmp_sample[tmp_sample<0.0001] = 0
            tmp_sample = tmp_sample/0.0008
            tmp_sample[tmp_sample>1] = 1
            tmp_shape = np.mean(tmp_sample, axis=-1)>0

            colors = np.zeros(tmp_shape.shape + (3,))
            colors[..., 0] = tmp_sample[..., 0]
            colors[..., 1] = tmp_sample[..., 2]
            colors[..., 2] = tmp_sample[..., 1]
            ax.voxels(tmp_shape, facecolors=colors, edgecolors=None, alpha=0.3)
            ax.set_xlim([-1, sample_dim+1])
            ax.set_ylim([-1, sample_dim+1])
            ax.set_zlim([-1, sample_dim+1])
            ax.set_aspect('equal')
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(f'{save_path}/plot_3d_{prefix}_{epoch:05d}.png', bbox_inches='tight')
    plt.close(fig)


plot_fn_dict = {"plot_cxdi_chart": plot_cxdi_chart,
                "plot_cxdi_NP": plot_cxdi_NP,
                "plot_xafs_3d": plot_xafs_3d}   