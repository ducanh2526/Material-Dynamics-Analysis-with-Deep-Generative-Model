import logging
import os
import pickle 
import glob
import time
import random

import tensorflow as tf
import numpy as np
import pandas as pd
import cv2 as cv 

from concurrent.futures import ThreadPoolExecutor, wait
from math import ceil
from scipy.ndimage import gaussian_filter, zoom
from skimage.measure import shannon_entropy
from itertools import product
from random import shuffle
from itertools import product

from tensorflow.keras.utils import Sequence

RNG_SEED = 123
logger = logging.getLogger(__name__)

def process_img(img, gaussian_sigma=None, 
                adaptive_size=None, adaptive_thresh=None,
                area_thresh=None):
    if gaussian_sigma is not None:
        filter_img = gaussian_filter(img, sigma=gaussian_sigma)
    else:
        filter_img = img.copy()
    if adaptive_size is not None:
        if adaptive_size%2==0:
            adaptive_size = int(adaptive_size//2+1)
        if len(img.shape)<3:
            filter_img = np.expand_dims(filter_img, axis=-1)
        uint_img = filter_img.astype(np.uint8)
        thresh = cv.adaptiveThreshold(uint_img, 255, cv.ADAPTIVE_THRESH_MEAN_C,\
                                    cv.THRESH_BINARY, adaptive_size, adaptive_thresh)
        
        mask_px = np.mean(img[thresh==255])
        img[thresh==255] = mask_px
    
        if area_thresh is not None:
            contours, _ = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
            cnt_surface = sorted(contours, key= lambda x: cv.contourArea(x),reverse=True)[1:]
            cnt_surface = [cnt for cnt in cnt_surface if cv.contourArea(cnt)>area_thresh]
            tmp_img = np.zeros(img.shape).astype(np.uint8)
            tmp_img = cv.cvtColor(tmp_img, cv.COLOR_GRAY2BGR)
            cv.drawContours(tmp_img, cnt_surface, -1, thickness=-1, color=(255, 255, 255))
            tmp_img = cv.cvtColor(tmp_img, cv.COLOR_BGR2GRAY)
            img[tmp_img==0] = mask_px
            thresh[tmp_img==0] = 255
    else:
        thresh = np.zeros(img.shape)

    return img, thresh

def get_txt(att_name, att, int_val=True):
    if len(np.array(att).shape)>=1:
        att = np.array(att).ravel().tolist()
    elif len(np.array(att).shape)==0:
        att = [att]
    att_txt = "|".join(str(int(e)) for e in att) if int_val else \
                "|".join(str(e) for e in att)
    return att_name + att_txt


class DataIterator(Sequence):
    """
    Create Data interator over dataset
    """

    def __init__(self, data_path, iter_type, 
                 resize_dim=None, 
                 add_info_dir=None, 
                 add_info=None, 
                 indices=None, 
                 dup_dims=None, 
                 n_jobs=1, batch_size=32):
        self.batch_size = batch_size
        self.iter_type = iter_type
        assert self.iter_type in ["train", "val", "test", "pred"]
        self.resize_dim = resize_dim
        if self.resize_dim is not None:
            assert len(self.resize_dim) in [2, 3]
        self.add_info_dir = add_info_dir 
        if self.add_info_dir is not None:
            assert os.path.isfile(self.add_info_dir)
            self.add_info = add_info
        else:
            self.add_info = None 
        self.dup_dims = dup_dims
        self.n_jobs = n_jobs

        self.data_path = data_path
        # Normalization term is derived in the condition that minimum image pixel is scaled to 0 
        norm_term_dir = self.data_path+"/norm_term.npy"
        if os.path.isfile(norm_term_dir):
            self.norm_term = np.load(norm_term_dir, allow_pickle=True)
        else:
            self.norm_term = None
        self.indices = indices 
        self.shuffle = True if (iter_type=='train') else False

        self.patch_names = [fn.replace(".npy", "") for fn in \
                            os.listdir(self.data_path) if ".npy" in fn]
        self.patch_names = [pn for pn in self.patch_names if not pn=="norm_term"]
        if self.indices is None:
            self.indices = np.arange(len(self.patch_names))
        self.n = len(self.indices)
        self.num_batch = ceil(self.n / self.batch_size)
        assert np.max(self.indices)<=len(self.patch_names)

        self.on_epoch_end()

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.indices))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __len__(self):
        return ceil(len(self.indices) / self.batch_size)

    def __getitem__(self, idx):
        indexes = self.indexes[idx * self.batch_size : (idx + 1) * self.batch_size]
        batch_imgs = []
        if self.add_info is not None:
            batch_prop = []
        for i in indexes:
            patch_name = self.patch_names[i]
            check_file = "{0}/{1}.npy".format(self.data_path, patch_name)
            with open(check_file, "rb") as handle:
                dat = np.load(handle, allow_pickle=True)
                batch_img = dat if len(dat)>2 else dat[0]
                if self.norm_term is not None:
                    batch_img = (batch_img-self.norm_term)/self.norm_term
                if len(batch_img.shape)<4:
                    batch_img = np.expand_dims(batch_img, axis=-1)
                if self.resize_dim is not None:
                    if len(self.resize_dim)==2:
                        batch_img = tf.image.resize(batch_img, 
                                                    self.resize_dim, 
                                                    method="bicubic")
                    else:
                        resize_ratio = np.array(self.resize_dim)/batch_img.shape[0]
                        batch_img = zoom(batch_img, zoom=list(resize_ratio)+[1])
                # assert batch_img.max()<=1 
                if self.dup_dims is not None:
                    batch_img = np.repeat(batch_img, self.dup_dims, -1)
            batch_imgs += [batch_img]

            if self.add_info is not None:
                patch_idx = int(patch_name.split("_")[1])
                with open(self.add_info_dir, "rb") as handle:
                    add_info_dict = pickle.load(handle)
                    assert self.add_info in list(add_info_dict.keys())
                    add_info_prop = add_info_dict[self.add_info][patch_idx]
                batch_prop += [add_info_prop]
            
        batch_imgs = tf.cast(np.array(batch_imgs), dtype=float)
        if self.add_info is not None:
            batch_prop = tf.cast(np.array(batch_prop), dtype=float)
        if self.add_info is not None:
            return batch_imgs, batch_prop
        else:
            return batch_imgs



        

