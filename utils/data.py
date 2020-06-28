# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 03:17:28 2020

@author: prnvb
"""

import random
import numpy as np

def onehotencode(a,wgan=True):
    a = a.reshape((a.shape[0],))
    if wgan:
        b = np.ones((a.size, a.max()+1),dtype=np.int32)*-1
    else:
        b = np.zeros((a.size, a.max()+1),dtype=np.int32)
    b[np.arange(a.size),a] = 1
    return b

def sample_known_unknown_classes(n_classes, n_known):
    classes = list(range(n_classes))
    known_classes = random.sample(classes,n_known)
    unknown_classes = list(set(classes)-set(known_classes))
    return classes, known_classes, unknown_classes

def sample_mismatch_labels(labels,known_classes):
    labels = labels.reshape((labels.shape[0],))
    mismatch_labels = []
    for l in labels:
        mismatch_labels.append(random.choice(
                list(filter(lambda x: x != l,known_classes))))
    mismatch_labels = np.array(mismatch_labels,dtype=np.int)
    return mismatch_labels.reshape(labels.shape)

def sample_mismatch_images(classwise,labels):
    labels = labels.reshape((labels.shape[0],))
    mismatch_images = []
    for l in labels:
        mismatch_images.append( random.choice(classwise[l]) )
    return np.array(mismatch_images, dtype=classwise[labels[0]].dtype)

def get_classwise(x,y,classes):
    y = y.reshape((y.shape[0],))
    classwise = {}
    for i in classes:
        classwise[i] = x[np.where(y==i)]
    return classwise

def get_mapped_labels(labels,mapping):
    labels = labels.reshape((labels.shape[0],))
    mapped_labels = []
    for i in range(len(labels)):
        mapped_labels.append(mapping[labels[i]])
    mapped_labels = np.array(mapped_labels,dtype=labels.dtype)
    return mapped_labels.reshape(labels.shape)
