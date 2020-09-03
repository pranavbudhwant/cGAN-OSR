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

def get_mismatch_data(x_known_classwise,y_known,known_classes,known_class_mapping):
    #Sample mismatch labels
    y_known_mismatch = sample_mismatch_labels(y_known,known_classes)
    #Sample mismatch images
    x_known_mismatch = sample_mismatch_images(x_known_classwise,y_known_mismatch)
    #Generate mapped labels
    y_known_mismatch_mapped = get_mapped_labels(y_known_mismatch,known_class_mapping)
    #Onehotencode mapped labels
    y_known_mismatch_mapped_ohe = onehotencode(y_known_mismatch_mapped+1,wgan=True)
    
    #Reshape arrays
    y_known_mismatch = y_known_mismatch.reshape((y_known_mismatch.shape[0],1))
    y_known_mismatch_mapped = y_known_mismatch_mapped.reshape((y_known_mismatch_mapped.shape[0],1))
    if len(x_known_mismatch.shape)<4:
        x_known_mismatch = np.expand_dims(x_known_mismatch,axis=-1)
    
    return (x_known_mismatch, y_known_mismatch, y_known_mismatch_mapped, y_known_mismatch_mapped_ohe)

def get_known_unknown_data(x,y,known_classes,known_class_mapping,unknown_classes,unknown=False):
    #Known - Match
    y_known = y[np.where(np.isin(y,known_classes))]
    x_known = x[np.where(np.isin(y,known_classes))]
    
    x_known_classwise = get_classwise(x_known,y_known,known_classes)
    
    #Generate mapped labels
    y_known_mapped = get_mapped_labels(y_known,known_class_mapping)
    #Onehotencode mapped labels
    y_known_mapped_ohe = onehotencode(y_known_mapped+1,wgan=True)
    
    #Reshape arrays
    y_known = y_known.reshape((y_known.shape[0],1))
    y_known_mapped = y_known_mapped.reshape((y_known_mapped.shape[0],1))
    if len(x_known.shape)<4:
        x_known = np.expand_dims(x_known,axis=-1)
    
    #Known - Mismatch
    x_known_mismatch, y_known_mismatch,\
    y_known_mismatch_mapped, y_known_mismatch_mapped_ohe = get_mismatch_data(x_known_classwise,
                                                                             y_known,known_classes,
                                                                             known_class_mapping)
    #Unknown
    if unknown:
        #y_unknown = y[np.where(np.isin(y,unknown_classes))]
        x_unknown = x[np.where(np.isin(y,unknown_classes))]

        if len(x_unknown.shape)<4:
            x_unknown = np.expand_dims(x_unknown,axis=-1)
            
        return (x_known, x_known_classwise, y_known, y_known_mapped, y_known_mapped_ohe), \
                (x_known_mismatch, y_known_mismatch, y_known_mismatch_mapped, y_known_mismatch_mapped_ohe),\
                (x_unknown)

    return (x_known, x_known_classwise, y_known, y_known_mapped, y_known_mapped_ohe), \
            (x_known_mismatch, y_known_mismatch, y_known_mismatch_mapped, y_known_mismatch_mapped_ohe)