# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 14:45:29 2020

@author: prnvb
"""

from tensorflow.keras import regularizers, initializers, constraints
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer, Input, InputSpec
from tensorflow.keras.models import Model
import tensorflow as tf

import random
import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import BatchNormalization, Conv2D, Dense, Flatten

class ConditionalAffine(Layer):
    def __init__(self,num_classes=1,
                 axis=-1,
                 beta_initializer='zeros',
                 gamma_initializer='ones',
                 beta_regularizer=None,
                 gamma_regularizer=None,
                 beta_constraint=None,
                 gamma_constraint=None,
                 **kwargs):
        self.axis = axis
        self.num_classes = num_classes
        self.beta_initializer = initializers.get(beta_initializer)
        self.gamma_initializer = initializers.get(gamma_initializer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        self.beta_constraint = constraints.get(beta_constraint)
        self.gamma_constraint = constraints.get(gamma_constraint)
        super(ConditionalAffine, self).__init__(**kwargs)
        
    def build(self, input_shape):
        dim = input_shape[0][self.axis]
        if dim is None:
            raise ValueError('Axis ' + str(self.axis) + ' of '
                             'input tensor should have a defined dimension '
                             'but the layer received an input with shape ' +
                             str(input_shape[0]) + '.')
        self.input_spec = [InputSpec(ndim=len(input_shape[0]),
                                    axes={self.axis: dim}),
                            InputSpec(ndim=len(input_shape[1]))]
        shape = (dim,)
        
        self.gamma = [self.add_weight(shape=shape,
                                     name='gamma_'+str(i),
                                     initializer=self.gamma_initializer,
                                     regularizer=self.gamma_regularizer,
                                     constraint=self.gamma_constraint)
                        for i in range(self.num_classes)]
        
        self.beta = [self.add_weight(shape=shape,
                                    name='beta_'+str(i),
                                    initializer=self.beta_initializer,
                                    regularizer=self.beta_regularizer,
                                    constraint=self.beta_constraint)
                        for i in range(self.num_classes)]
        
        self.built = True
        
    def call(self, inputs):
        input_shape = K.int_shape(inputs[0])
        in_shape = K.shape(inputs[0])
        input_class = tf.cast(inputs[1], tf.int32)
        dim = input_shape[self.axis]
        
        #Only supports NHWC
        ndim = len(input_shape)
        shape = tf.stack( [in_shape[0]] +[1]*(ndim-2) + [dim] )
        
        gamma = tf.stack(values=self.gamma, axis=0)
        beta = tf.stack(values=self.beta, axis=0)

        _gamma = K.reshape(tf.gather(params=gamma,indices=input_class,axis=0),shape)
        _beta = K.reshape(tf.gather(params=beta,indices=input_class,axis=0),shape)
        
        scaled_shifted = inputs[0]*_gamma + _beta
        
        return scaled_shifted
        
    def compute_output_shape(self, input_shape):
        return input_shape[0]

    #If not overridden, throws "NotImplementedError: Layer ConditionalAffine has arguments in `__init__` and therefore must override `get_config`."
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'axis': self.axis,
            'num_classes': self.num_classes,
            'beta_initializer': self.beta_initializer,
            'gamma_initializer': self.gamma_initializer,
            'beta_regularizer': self.beta_regularizer,
            'gamma_regularizer': self.gamma_regularizer,
            'beta_constraint': self.beta_constraint,
            'gamma_constraint': self.gamma_constraint
        })
        return config

class ConditionalBatchNormalization(Layer):
    def __init__(self,
                 axis=-1,
                 momentum=0.99,
                 epsilon=1e-3,
                 center=True,
                 scale=True,
                 beta_initializer='zeros',
                 gamma_initializer='ones',
                 moving_mean_initializer='zeros',
                 moving_variance_initializer='ones',
                 beta_regularizer=None,
                 gamma_regularizer=None,
                 beta_constraint=None,
                 gamma_constraint=None,
                 num_classes=1,
                 **kwargs):
        super(ConditionalBatchNormalization, self).__init__(**kwargs)
        self.supports_masking = False
        self.axis = axis
        self.momentum = momentum
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.beta_initializer = initializers.get(beta_initializer)
        self.gamma_initializer = initializers.get(gamma_initializer)
        self.moving_mean_initializer = initializers.get(moving_mean_initializer)
        self.moving_variance_initializer = (
            initializers.get(moving_variance_initializer))
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        self.beta_constraint = constraints.get(beta_constraint)
        self.gamma_constraint = constraints.get(gamma_constraint)
        self.num_classes = num_classes

    def build(self, input_shape):
        dim = input_shape[0][self.axis]
        #print(input_shape)
        if dim is None:
            raise ValueError('Axis ' + str(self.axis) + ' of '
                             'input tensor should have a defined dimension '
                             'but the layer received an input with shape ' +
                             str(input_shape[0]) + '.')
        self.input_spec = [InputSpec(ndim=len(input_shape[0]),
                                    axes={self.axis: dim}),
                            InputSpec(ndim=len(input_shape[1]))]
        shape = (dim,)

        if self.scale:
            self.gamma = [self.add_weight(shape=shape,
                                         name='gamma_'+str(i),
                                         initializer=self.gamma_initializer,
                                         regularizer=self.gamma_regularizer,
                                         constraint=self.gamma_constraint)
                            for i in range(self.num_classes)]
        else:
            self.gamma = [None for i in range(self.num_classes)]
        if self.center:
            self.beta = [self.add_weight(shape=shape,
                                        name='beta_'+str(i),
                                        initializer=self.beta_initializer,
                                        regularizer=self.beta_regularizer,
                                        constraint=self.beta_constraint)
                            for i in range(self.num_classes)]
        else:
            self.beta = [None for i in range(self.num_classes)]
            
        self.moving_mean = self.add_weight(
            shape=shape,
            name='moving_mean',
            initializer=self.moving_mean_initializer,
            trainable=False)
        self.moving_variance = self.add_weight(
            shape=shape,
            name='moving_variance',
            initializer=self.moving_variance_initializer,
            trainable=False)
        self.built = True

    def call(self, inputs, training=None):
        input_shape = K.int_shape(inputs[0])
        in_shape = K.shape(inputs[0])
        input_class = inputs[1]
        dim = input_shape[self.axis]
        
        #Only supports NHWC
        ndim = len(input_shape)
        shape = tf.stack( [in_shape[0]] +[1]*(ndim-2) + [dim] )
        
        '''
        if len(input_shape)==2:
            shape = tf.stack([in_shape[0],dim])
        elif len(input_shape)==4:
            shape = tf.stack([in_shape[0],1,1,dim])
        '''
        
        #print(input_shape)
        #print(input_class)
        #print( tf.gather(self.beta,input_class) )
        #print( tf.squeeze(tf.gather(self.beta,input_class)) )
        #print( K.reshape(tf.gather(self.beta,input_class),shape) )
        #print( K.reshape(tf.gather(self.beta,input_class),shape) )
        
        # Prepare broadcasting shape.
        ndim = len(input_shape)
        reduction_axes = list(range(len(input_shape)))
        del reduction_axes[self.axis]
        broadcast_shape = [1] * len(input_shape)
        broadcast_shape[self.axis] = input_shape[self.axis]
        
        #print(broadcast_shape)
        
        # Determines whether broadcasting is needed.
        needs_broadcasting = (sorted(reduction_axes) != list(range(ndim))[:-1])

        def normalize_inference():
            if needs_broadcasting:
                # In this case we must explicitly broadcast all parameters.
                broadcast_moving_mean = K.reshape(self.moving_mean,
                                                  broadcast_shape)
                broadcast_moving_variance = K.reshape(self.moving_variance,
                                                      broadcast_shape)
                if self.center:
                    broadcast_beta = K.reshape(tf.gather(self.beta,
                                                         input_class),shape)
                                               #K.reshape(,broadcast_shape)
                else:
                    broadcast_beta = None
                    
                if self.scale:
                    broadcast_gamma = K.reshape(tf.gather(self.gamma,
                                                          input_class),shape)
                                                #K.reshape(,broadcast_shape)
                else:
                    broadcast_gamma = None
                
                norm = K.batch_normalization(
                    inputs[0],
                    broadcast_moving_mean,
                    broadcast_moving_variance,
                    None,
                    None,
                    axis=self.axis,
                    epsilon=self.epsilon)
                if self.scale:
                    norm *= broadcast_gamma
                if self.center:
                    norm += broadcast_beta
                return norm
            else:
                norm = K.batch_normalization(
                    inputs[0],
                    self.moving_mean,
                    self.moving_variance,
                    None,
                    None,
                    axis=self.axis,
                    epsilon=self.epsilon)
                if self.scale:
                    norm *= K.reshape(tf.gather(self.gamma,input_class),shape)
                if self.center:
                    norm += K.reshape(tf.gather(self.beta,input_class),shape)
                return norm

        # If the learning phase is *static* and set to inference:
        if training in {0, False}:
            #print('INFERENCE')
            return normalize_inference()

        # If the learning is either dynamic, or set to training:
        normed_training, mean, variance = K.normalize_batch_in_training(
            inputs[0], None, None, 
            reduction_axes,
            epsilon=self.epsilon)
        
        if self.scale:
            normed_training *= K.reshape(tf.gather(self.gamma,input_class),shape)
        if self.center:
            normed_training += K.reshape(tf.gather(self.beta,input_class),shape)
        
        if K.backend() != 'cntk':
            sample_size = K.prod([K.shape(inputs[0])[axis]
                                  for axis in reduction_axes])
            sample_size = K.cast(sample_size, dtype=K.dtype(inputs[0]))
            if K.backend() == 'tensorflow' and sample_size.dtype != 'float32':
                sample_size = K.cast(sample_size, dtype='float32')

            # sample variance - unbiased estimator of population variance
            variance *= sample_size / (sample_size - (1.0 + self.epsilon))

        self.add_update([K.moving_average_update(self.moving_mean,
                                                 mean,
                                                 self.momentum),
                         K.moving_average_update(self.moving_variance,
                                                 variance,
                                                 self.momentum)]) #,inputs[0]

        # Pick the normalized form corresponding to the training phase.
        return K.in_train_phase(normed_training,
                                normalize_inference,
                                training=training)

    def get_config(self):
        config = {
            'axis': self.axis,
            'momentum': self.momentum,
            'epsilon': self.epsilon,
            'center': self.center,
            'scale': self.scale,
            'num_classes':self.num_classes,
            'beta_initializer': initializers.serialize(self.beta_initializer),
            'gamma_initializer': initializers.serialize(self.gamma_initializer),
            'moving_mean_initializer':
                initializers.serialize(self.moving_mean_initializer),
            'moving_variance_initializer':
                initializers.serialize(self.moving_variance_initializer),
            'beta_regularizer': regularizers.serialize(self.beta_regularizer),
            'gamma_regularizer': regularizers.serialize(self.gamma_regularizer),
            'beta_constraint': constraints.serialize(self.beta_constraint),
            'gamma_constraint': constraints.serialize(self.gamma_constraint)
        }
        base_config = super(ConditionalBatchNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape[0]
    '''
    def compute_mask(self, inputs, mask=None):
        # Just pass the received mask from previous layer, to the next 
        #layer or 
        # manipulate it if this layer changes the shape of the input
        return mask[1]
    '''


if __name__ == '__main__':
    
    input_image = Input(shape=(28,28,1))
    c = Input(shape=(1,),dtype=tf.int32)
    x = Conv2D(filters=32,kernel_size=3,strides=2,padding='same')(input_image)
    x = BatchNormalization(scale=False,center=False)(x)
    x = ConditionalAffine(num_classes=3)([x,c])
    x = Flatten()(x)
    x = Dense(10,activation='linear')(x)
    model = Model([input_image,c],x)
    
    #model.summary()
    
    model.compile(optimizer='adam',loss='mse')
    
    X = np.random.rand(32,28,28,1)
    Y = np.random.rand(32,10)
    C = np.ones((32,1))*2
    
    weights_before = model.layers[4].get_weights()
    model.train_on_batch(x=[X, C], y=Y)
    weights_after = model.layers[4].get_weights()
    
    
    
    x = Input((10,))
    c = Input(shape=(1,),dtype=tf.int32)
    h = ConditionalBatchNormalization(num_classes=3)([x, c])
    model = Model([x, c], h)
    model.summary()
    model.compile(optimizer=Adam(1e-4), loss='mse')
    
    #C = np.array([random.randint(0, 2) for i in range(100)])
    C = np.ones((100,1))*0
    
    X = np.random.rand(100, 10)
    Y = np.random.rand(100, 10)
    
    weights_before = model.layers[2].get_weights()
    model.train_on_batch(x=[X, C], y=Y)
    weights_after = model.layers[2].get_weights()
    
    res=model.predict_on_batch(x=[X,C])
    
    x = Input((10,))
    h = BatchNormalization()(x)
    model = Model(x, h)
    model.summary()
    model.compile(optimizer=Adam(1e-4), loss='mse')
    
    X = np.random.rand(100, 10)
    Y = np.random.rand(100, 10)
    weights_before = model.layers[1].get_weights()
    model.train_on_batch(x=X, y=Y)
    weights_after = model.layers[1].get_weights()
    
    
    
    
    
'''
global c1, c2, c3
c1 = K.variable([0])
c2 = K.variable([0])
c3 = K.variable([0])

class ConditionalBatchNormalization(Layer):
    """Conditional Batch normalization layer."""
    @interfaces.legacy_batchnorm_support
    def __init__(self, 
                 axis=-1,
                 momentum=0.99,
                 epsilon=1e-3,
                 center=True,
                 scale=True,
                 beta_initializer='zeros',
                 gamma_initializer='ones',
                 moving_mean_initializer='zeros',
                 moving_variance_initializer='ones',
                 beta_regularizer=None,
                 gamma_regularizer=None,
                 beta_constraint=None,
                 gamma_constraint=None,
                 **kwargs):
        super(ConditionalBatchNormalization, self).__init__(**kwargs)
        self.axis = axis
        self.momentum = momentum
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.beta_initializer = initializers.get(beta_initializer)
        self.gamma_initializer = initializers.get(gamma_initializer)
        self.moving_mean_initializer = initializers.get(moving_mean_initializer)
        self.moving_variance_initializer = (
            initializers.get(moving_variance_initializer))
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        self.beta_constraint = constraints.get(beta_constraint)
        self.gamma_constraint = constraints.get(gamma_constraint)
    
    
    def build(self, input_shape):
    
        dim = input_shape[0][self.axis]
        if dim is None:
            raise ValueError('Axis ' + str(self.axis) + ' of '
                             'input tensor should have a defined dimension '
                             'but the layer received an input with shape ' +
                             str(input_shape[0]) + '.')
    
        shape = (dim,)
    
        if self.scale:
            self.gamma1 = self.add_weight(shape=shape,
                                         name='gamma',
                                         initializer=self.gamma_initializer,
                                         regularizer=self.gamma_regularizer,
                                         constraint=self.gamma_constraint)
            self.gamma2 = self.add_weight(shape=shape,
                                         name='gamma',
                                         initializer=self.gamma_initializer,
                                         regularizer=self.gamma_regularizer,
                                         constraint=self.gamma_constraint)
            self.gamma3 = self.add_weight(shape=shape,
                                         name='gamma',
                                         initializer=self.gamma_initializer,
                                         regularizer=self.gamma_regularizer,
                                         constraint=self.gamma_constraint)
        else:
            self.gamma1 = None
            self.gamma2 = None
            self.gamma3 = None
    
        if self.center:
            self.beta1 = self.add_weight(shape=shape,
                                        name='beta',
                                        initializer=self.beta_initializer,
                                        regularizer=self.beta_regularizer,
                                        constraint=self.beta_constraint)
    
            self.beta2 = self.add_weight(shape=shape,
                                        name='beta',
                                        initializer=self.beta_initializer,
                                        regularizer=self.beta_regularizer,
                                        constraint=self.beta_constraint)
    
            self.beta3 = self.add_weight(shape=shape,
                                        name='beta',
                                        initializer=self.beta_initializer,
                                        regularizer=self.beta_regularizer,
                                        constraint=self.beta_constraint)
        else:
            self.beta1 = None
            self.beta2 = None
            self.beta3 = None
    
        self.moving_mean = self.add_weight(
            shape=shape,
            name='moving_mean',
            initializer=self.moving_mean_initializer,
            trainable=False)
        self.moving_variance = self.add_weight(
            shape=shape,
            name='moving_variance',
            initializer=self.moving_variance_initializer,
            trainable=False)
    
        super(ConditionalBatchNormalization, self).build(input_shape) 
    
    def call(self, inputs, training=None):
    
        input_shape = K.int_shape(inputs[0])
        c1 = inputs[1][0]
        c2 = inputs[2][0]
    
        # Prepare broadcasting shape.
        ndim = len(input_shape)
        reduction_axes = list(range(len(input_shape)))
        del reduction_axes[self.axis]
        broadcast_shape = [1] * len(input_shape)
        broadcast_shape[self.axis] = input_shape[self.axis]
    
        # Determines whether broadcasting is needed.
        needs_broadcasting = (sorted(reduction_axes) != list(range(ndim))[:-1])
    
        def normalize_inference():
            if needs_broadcasting:
                # In this case we must explicitly broadcast all parameters.
                broadcast_moving_mean = K.reshape(self.moving_mean,
                                                  broadcast_shape)
                broadcast_moving_variance = K.reshape(self.moving_variance,
                                                      broadcast_shape)
                if self.center:
                    broadcast_beta = \
                        tf.case({
                                    c1: lambda: K.reshape(self.beta1,
                                                          broadcast_shape),
                                    c2: lambda: K.reshape(self.beta2,
                                                          broadcast_shape)
                                },
                                    default=lambda: K.reshape(self.beta3,
                                                              broadcast_shape)
                                )
    
                else:
                    broadcast_beta = None
    
                if self.scale:
    
                    broadcast_gamma = \
                        tf.case({
                                    c1: lambda: K.reshape(self.gamma1,
                                                          broadcast_shape),
                                    c2: lambda: K.reshape(self.gamma2,
                                                          broadcast_shape)
                                },
                                    default=lambda: K.reshape(self.gamma3,
                                                              broadcast_shape)
                                )
    
                else:
                    broadcast_gamma = None
    
                return K.batch_normalization(
                    inputs[0],
                    broadcast_moving_mean,
                    broadcast_moving_variance,
                    broadcast_beta,
                    broadcast_gamma,
                    axis=self.axis,
                    epsilon=self.epsilon)
            else:
                out = \
                tf.case({
                        c1: lambda: K.batch_normalization(
                                            inputs[0],
                                            self.moving_mean,
                                            self.moving_variance,
                                            self.beta1,
                                            self.gamma1,
                                            axis=self.axis,
                                            epsilon=self.epsilon),
                        c2: lambda: K.batch_normalization(
                                            inputs[0],
                                            self.moving_mean,
                                            self.moving_variance,
                                            self.beta2,
                                            self.gamma2,
                                            axis=self.axis,
                                            epsilon=self.epsilon)
                    },
                        default=lambda: K.batch_normalization(
                                            inputs[0],
                                            self.moving_mean,
                                            self.moving_variance,
                                            self.beta3,
                                            self.gamma3,
                                            axis=self.axis,
                                            epsilon=self.epsilon)
                            )
    
                return out
    
        # If the learning phase is *static* and set to inference:
        if training in {0, False}:
            return normalize_inference()
    
    
        # If the learning is either dynamic, or set to training:
        normed_training, mean, variance = \
            tf.case({
                        c1: lambda: K.normalize_batch_in_training(
                                inputs[0], self.gamma1, self.beta1, reduction_axes,
                                epsilon=self.epsilon),
                        c2: lambda: K.normalize_batch_in_training(
                                inputs[0], self.gamma2, self.beta2, reduction_axes,
                                epsilon=self.epsilon)
                    },
                        default=lambda: K.normalize_batch_in_training(
                                inputs[0], self.gamma3, self.beta3, reduction_axes,
                                epsilon=self.epsilon)
                    )
    
        print(normed_training)
    
        if K.backend() != 'cntk':
            sample_size = K.prod([K.shape(inputs[0])[axis]
                                  for axis in reduction_axes])
            sample_size = K.cast(sample_size, dtype=K.dtype(inputs[0]))
            if K.backend() == 'tensorflow' and sample_size.dtype != 'float32':
                sample_size = K.cast(sample_size, dtype='float32')
    
            # sample variance - unbiased estimator of population variance
            variance *= sample_size / (sample_size - (1.0 + self.epsilon))
    
        self.add_update([K.moving_average_update(self.moving_mean,
                                                 mean,
                                                 self.momentum),
                         K.moving_average_update(self.moving_variance,
                                                 variance,
                                                 self.momentum)],
                        inputs[0])
    
        # Pick the normalized form corresponding to the training phase.
    
        return K.in_train_phase(normed_training,
                                normalize_inference,
                                training=training)
    
    def get_config(self):
        config = {
            'axis': self.axis,
            'momentum': self.momentum,
            'epsilon': self.epsilon,
            'center': self.center,
            'scale': self.scale,
            'beta_initializer': initializers.serialize(self.beta_initializer),
            'gamma_initializer': initializers.serialize(self.gamma_initializer),
            'moving_mean_initializer':
                initializers.serialize(self.moving_mean_initializer),
            'moving_variance_initializer':
                initializers.serialize(self.moving_variance_initializer),
            'beta_regularizer': regularizers.serialize(self.beta_regularizer),
            'gamma_regularizer': regularizers.serialize(self.gamma_regularizer),
            'beta_constraint': constraints.serialize(self.beta_constraint),
            'gamma_constraint': constraints.serialize(self.gamma_constraint)
        }
        base_config = super(ConditionalBatchNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
    def compute_output_shape(self, input_shape):
    
        return input_shape[0]
'''
