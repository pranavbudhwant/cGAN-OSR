from keras.layers import Input, Dense, Conv2D, Add, Dot, Conv2DTranspose,\
 Activation, Reshape,BatchNormalization,UpSampling2D,AveragePooling2D, \
 GlobalAveragePooling2D, LeakyReLU, Flatten, Concatenate, Embedding
from keras.models import Model, Sequential
import keras.backend as K
from keras.utils import plot_model
from SpectralNormalizationKeras import DenseSN, ConvSN2D, EmbeddingSN
from keras.layers.pooling import _GlobalPooling2D

from CBN import ConditionalAffine

class GlobalSumPooling2D(_GlobalPooling2D):
    """Global sum pooling operation for spatial data.
    # Arguments
        data_format: A string,
            one of `channels_last` (default) or `channels_first`.
            The ordering of the dimensions in the inputs.
            `channels_last` corresponds to inputs with shape
            `(batch, height, width, channels)` while `channels_first`
            corresponds to inputs with shape
            `(batch, channels, height, width)`.
            It defaults to the `image_data_format` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "channels_last".
    # Input shape
        - If `data_format='channels_last'`:
            4D tensor with shape:
            `(batch_size, rows, cols, channels)`
        - If `data_format='channels_first'`:
            4D tensor with shape:
            `(batch_size, channels, rows, cols)`
    # Output shape
        2D tensor with shape:
        `(batch_size, channels)`
    """

    def call(self, inputs):
        if self.data_format == 'channels_last':
            return K.sum(inputs, axis=[1, 2])
        else:
            return K.sum(inputs, axis=[2, 3])


def ResBlock(input_shape, sampling=None, trainable_sortcut=True, 
             spectral_normalization=False, batch_normalization=True,
             bn_momentum=0.9, bn_epsilon=0.00002, cbn=0,
             channels=256, k_size=3, summary=False,
             plot=False, name=None):
    '''
    ResBlock(input_shape, sampling=None, trainable_sortcut=True, 
             spectral_normalization=False, batch_normalization=True,
             bn_momentum=0.9, bn_epsilon=0.00002,
             channels=256, k_size=3, summary=False,
             plot=False, plot_name='res_block.png')""
             
    Build ResBlock as keras Model
    sampleing = 'up' for upsampling
                'down' for downsampling(AveragePooling)
                None for none
    
    '''
    #input_shape = input_layer.sahpe.as_list()
    
    res_block_input = Input(shape=input_shape)
    if cbn:
        res_block_class_input = Input(shape=(1,),dtype='int32')
    
    
    if cbn:
        res_block_1 = BatchNormalization(momentum=bn_momentum, 
                                         epsilon=bn_epsilon,
                                     scale=False,center=False)(res_block_input)
        res_block_1 = ConditionalAffine(num_classes=cbn)([res_block_1,
                                       res_block_class_input])
    elif batch_normalization:
        res_block_1 = BatchNormalization(momentum=bn_momentum, 
                                         epsilon=bn_epsilon)(res_block_input)
    else:
        res_block_1 = res_block_input
        
    res_block_1     = Activation('relu')(res_block_1)
    
    if spectral_normalization:
        res_block_1     = ConvSN2D(channels, k_size , strides=1, padding='same',
                           kernel_initializer='glorot_uniform')(res_block_1)
    else:
        res_block_1     = Conv2D(channels, k_size , strides=1, padding='same',
                             kernel_initializer='glorot_uniform')(res_block_1)
    
    if sampling=='up':
        res_block_1     = UpSampling2D()(res_block_1)
    else:
        pass
    
    if cbn:
        res_block_2 = BatchNormalization(momentum=bn_momentum, 
                                         epsilon=bn_epsilon,
                                         scale=False,center=False)(res_block_1)
        res_block_2 = ConditionalAffine(num_classes=cbn)([res_block_2,
                                       res_block_class_input])
    elif batch_normalization:
        res_block_2     = BatchNormalization(momentum=bn_momentum, 
                                             epsilon=bn_epsilon)(res_block_1)
    else:
        res_block_2     = res_block_1
    res_block_2     = Activation('relu')(res_block_2)
    
    if spectral_normalization:
        res_block_2     = ConvSN2D(channels, k_size , strides=1, padding='same',
                             kernel_initializer='glorot_uniform')(res_block_2)
    else:
        res_block_2     = Conv2D(channels, k_size , strides=1, padding='same',
                             kernel_initializer='glorot_uniform')(res_block_2)
    
    if sampling=='down':
        res_block_2 = AveragePooling2D()(res_block_2)
    else:
        pass
    
    if trainable_sortcut:
        if spectral_normalization:
            short_cut = ConvSN2D(channels, 1 , strides=1, padding='same',
                         kernel_initializer='glorot_uniform')(res_block_input)
        else:
            short_cut = Conv2D(channels, 1 , strides=1, padding='same',
                       kernel_initializer='glorot_uniform')(res_block_input)
    else:
        short_cut = res_block_input
        
    if sampling=='up':
        short_cut       = UpSampling2D()(short_cut)
    elif sampling=='down':
        short_cut       = AveragePooling2D()(short_cut)
    elif sampling=='None':
        pass

    res_block_add   = Add()([short_cut, res_block_2])
    
    if cbn:
        res_block = Model([res_block_input,res_block_class_input], 
                          res_block_add, name=name)
    else:
        res_block = Model(res_block_input, res_block_add, name=name)
    
    if plot:
        plot_model(res_block, name+'.png', show_layer_names=False)
    if summary:
        print(name)
        res_block.summary()
    
    return res_block
    
def BuildGenerator(summary=True, bn_momentum=0.9, noise=False,
                   init_shape=(4,4,256), out_channels=3,
                   resblock3 = True, in_shape = (128,),
                   bn_epsilon=0.00002, cbn=0, spectral_normalization=False,
                   name='Generator', plot=False):
    
    conv2D = Conv2D
    dense = Dense
    if spectral_normalization:
        conv2D = ConvSN2D
        dense = DenseSN
        
    model_input = Input(shape=in_shape)
    if noise:
        model_input_noise = Input(shape=in_shape)
    if cbn:
        input_class = Input(shape=(1,),dtype='int32')
    
    dims = 1
    for s in init_shape:
        dims *= s
    
    x = model_input
    if noise:
        x = Concatenate()([model_input,model_input_noise])
    
    h = dense(dims, kernel_initializer='glorot_uniform')(x)
    h = Reshape(init_shape)(h)
    
    resblock_1  = ResBlock(input_shape=init_shape, sampling='up', 
                           spectral_normalization=spectral_normalization,
                           bn_epsilon=bn_epsilon, 
                           bn_momentum=bn_momentum, cbn=cbn, 
                           name='Generator_resblock_1')
    if cbn:
        h = resblock_1([h,input_class])
    else:
        h = resblock_1(h)
        
    s2 = init_shape[0]*2
    resblock_2  = ResBlock(input_shape=(s2,s2,256), sampling='up', 
                           spectral_normalization=spectral_normalization,
                           bn_epsilon=bn_epsilon, bn_momentum=bn_momentum, 
                           cbn=cbn, name='Generator_resblock_2')
    if cbn:
        h = resblock_2([h,input_class])
    else:
        h = resblock_2(h)
        
        
    if resblock3:
        s3 = init_shape[0]*2*2
        resblock_3  = ResBlock(input_shape=(s3,s3,256), sampling='up', 
                               spectral_normalization=spectral_normalization,
                               bn_epsilon=bn_epsilon, bn_momentum=bn_momentum, 
                               cbn=cbn, name='Generator_resblock_3')
        if cbn:
            h = resblock_3([h,input_class])
        else:
            h = resblock_3(h)
        
    if cbn:
        h = BatchNormalization(epsilon=bn_epsilon, momentum=bn_momentum,
                               center=False,scale=False)(h)
        h = ConditionalAffine(num_classes=cbn)([h,input_class])
    else:
        h = BatchNormalization(epsilon=bn_epsilon, momentum=bn_momentum)(h)
        
    h = Activation('relu')(h)
    
    model_output= conv2D(out_channels, kernel_size=3, strides=1, 
                         padding='same', activation='tanh')(h)
    
    inputs = [model_input]
    if noise:
        inputs += [model_input_noise]
    if cbn:
        inputs += [input_class]
        
    model = Model(inputs, model_output, name=name)
        
    if plot:
        plot_model(model, name+'.png', show_layer_names=True)
    if summary:
        print("Generator")
        print('Spectral Normalization: {}'.format(spectral_normalization))
        model.summary()
        
    return model

def BuildDiscriminator(summary=True, in_shape = (32,32,3),
                       spectral_normalization=True, 
                       batch_normalization=False, bn_momentum=0.9, 
                       bn_epsilon=0.00002, cbn=0,
                       name='Discriminator', 
                       plot=False):
    
    dense = Dense
    embedding = Embedding
    if spectral_normalization:
        dense = DenseSN
        embedding = EmbeddingSN
    
    if cbn:
        # label input
        in_label = Input(shape=(1,))
        # embedding for categorical input
        li = embedding(cbn, 50)(in_label)
        # scale up to image dimensions with linear activation
        n_nodes = in_shape[0] * in_shape[1]
        li = dense(n_nodes)(li)
        # reshape to additional channel
        li = Reshape((in_shape[0], in_shape[1], 1))(li)
        # image input
        in_image = Input(shape=in_shape)
        # concat label as a channel
        merge = Concatenate()([in_image, li])
        x = merge
        inp_shape = list(in_shape)
        inp_shape[-1] += 1
        resblock_1  = ResBlock(input_shape=tuple(inp_shape), channels=128, 
                               sampling='down', batch_normalization=True, 
                               spectral_normalization=spectral_normalization,
                               name='Discriminator_resblock_Down_1')
    else:
        model_input = Input(shape=in_shape)
        x = model_input
        resblock_1  = ResBlock(input_shape=in_shape, channels=128, 
                               sampling='down', batch_normalization=True, 
                               spectral_normalization=spectral_normalization,
                               name='Discriminator_resblock_Down_1')
        
    h           = resblock_1(x)
    s2 = int(in_shape[0]/2)
    resblock_2  = ResBlock(input_shape=(s2,s2,128),channels=128, 
                           sampling='down', batch_normalization=True, 
                           spectral_normalization=spectral_normalization, 
                           name='Discriminator_resblock_Down_2')
    h           = resblock_2(h)
    s3 = int(in_shape[0]/4)
    resblock_3  = ResBlock(input_shape=(s3,s3,128),channels=128, 
                           sampling=None, batch_normalization=True, 
                           spectral_normalization=spectral_normalization, 
                           trainable_sortcut=False, 
                           name='Discriminator_resblock_1' )
    h           = resblock_3(h)
    resblock_4  = ResBlock(input_shape=(s3,s3,128),channels=128, 
                           sampling=None, batch_normalization=True, 
                           spectral_normalization=spectral_normalization, 
                           trainable_sortcut=False, 
                           name='Discriminator_resblock_2' )
    h           = resblock_4(h)
    h           = Activation('relu')(h)
    h           = GlobalSumPooling2D()(h)
    model_output= dense(1,kernel_initializer='glorot_uniform')(h)
    
    if cbn:
        model = Model([in_image,in_label], model_output, name=name)
    else:
        model = Model(model_input, model_output, name=name)
        
    if plot:
        plot_model(model, name+'.png', show_layer_names=True)
        
    if summary:
        print('Discriminator')
        print('Spectral Normalization: {}'.format(spectral_normalization))
        model.summary()
    return model

def BuildDiscriminatorCS(summary=True, in_shape = (32,32,3),
                       spectral_normalization=True, 
                       batch_normalization=False, bn_momentum=0.9, 
                       bn_epsilon=0.00002, name='Discriminator', 
                       num_classes=10,feat=False,
                       plot=False):
    
    dense = Dense
    if spectral_normalization:
        dense = DenseSN
    
    model_input = Input(shape=in_shape)
    resblock_1  = ResBlock(input_shape=in_shape, channels=128, 
                           sampling='down', batch_normalization=True, 
                           spectral_normalization=spectral_normalization,
                           name='Discriminator_resblock_Down_1')
        
    h           = resblock_1(model_input)
    s2 = int(in_shape[0]/2)
    resblock_2  = ResBlock(input_shape=(s2,s2,128),channels=128, 
                           sampling='down', batch_normalization=True, 
                           spectral_normalization=spectral_normalization, 
                           name='Discriminator_resblock_Down_2')
    h           = resblock_2(h)
    s3 = int(in_shape[0]/4)
    resblock_3  = ResBlock(input_shape=(s3,s3,128),channels=128, 
                           sampling=None, batch_normalization=True, 
                           spectral_normalization=spectral_normalization, 
                           trainable_sortcut=False, 
                           name='Discriminator_resblock_1' )
    h           = resblock_3(h)
    resblock_4  = ResBlock(input_shape=(s3,s3,128),channels=128, 
                           sampling=None, batch_normalization=True, 
                           spectral_normalization=spectral_normalization, 
                           trainable_sortcut=False, 
                           name='Discriminator_resblock_2' )
    h           = resblock_4(h)
    h           = Activation('relu')(h)
    h           = GlobalSumPooling2D()(h)
    interm_output = h
    model_output= dense(num_classes+1,kernel_initializer='glorot_uniform')(h)
    
    if feat:
        model = Model(model_input, model_output, name=name)
        model_feat = Model(model_input, interm_output, name=name+'_feat')
    else:
        model = Model(model_input, model_output, name=name)
        
    if plot:
        plot_model(model, name+'.png', show_layer_names=True)
        if feat:
            plot_model(model_feat, name+'_feat.png', show_layer_names=True)
        
    if summary:
        print('Discriminator')
        print('Spectral Normalization: {}'.format(spectral_normalization))
        model.summary()
        if feat:
            model_feat.summary()
            
    if feat:
        return model, model_feat
    return model


def BuildEncoder(summary=True, in_shape = (32,32,3),
                 name='Encoder',latent_dim = 128, plot=False):
    
    model_input = Input(shape=in_shape)
    resblock_1  = ResBlock(input_shape=in_shape, channels=128, 
                           sampling='down', batch_normalization=True, 
                           spectral_normalization=False,
                           name='Discriminator_resblock_Down_1')
        
    h           = resblock_1(model_input)
    s2 = int(in_shape[0]/2)
    resblock_2  = ResBlock(input_shape=(s2,s2,128),channels=128, 
                           sampling='down', batch_normalization=True, 
                           spectral_normalization=False, 
                           name='Discriminator_resblock_Down_2')
    h           = resblock_2(h)
    s3 = int(in_shape[0]/4)
    resblock_3  = ResBlock(input_shape=(s3,s3,128),channels=128, 
                           sampling=None, batch_normalization=True, 
                           spectral_normalization=False, 
                           trainable_sortcut=False, 
                           name='Discriminator_resblock_1' )
    h           = resblock_3(h)
    resblock_4  = ResBlock(input_shape=(s3,s3,128),channels=latent_dim, 
                           sampling=None, batch_normalization=True, 
                           spectral_normalization=False, 
                           trainable_sortcut=False, 
                           name='Discriminator_resblock_2' )
    h           = resblock_4(h)
    h           = Activation('relu')(h)
    encoder_output = GlobalSumPooling2D()(h)
    
    encoder = Model(model_input, encoder_output, name=name)
    
    if plot:
        plot_model(encoder, name+'.png', show_layer_names=True)
        
    if summary:
        print('Encoder')
        encoder.summary()
        
    return encoder

def BuildClassifier(encoder, summary=True, in_shape = (32,32,3),
                    name='Classifier', num_classes = 10, plot=False):
    
    model_input = Input(shape=in_shape)
    encoder_output = encoder(model_input)
    
    classifier_output= Dense(num_classes,
                             kernel_initializer='glorot_uniform',
                             activation='softmax')(encoder_output)
    
    classifier = Model(model_input, classifier_output, name=name)
    
    if plot:
        plot_model(classifier, name+'.png', show_layer_names=True)
    
    if summary:
        print('Classifier')
        classifier.summary()
        
    return classifier


if __name__ == '__main__':
    print('Plot the model visualization')
    DIR = 'img/model/'
    
    print('DCGAN_Generator')
    model = BuildGenerator(resnet=False)
    plot_model(model, show_shapes=True, to_file=DIR+'DCGAN_Generator.png')
    
    
    print('ResNet_Generator')
    #MNIST
    model = BuildGenerator(cbn=10,resblock3=False,
                           spectral_normalization=True,out_channels=1,
                           init_shape=(7,7,256))

    #CIFAR
    model = BuildGenerator(cbn=10, spectral_normalization=True)
    plot_model(model, show_shapes=True, to_file=DIR+'ResNet_Generator.png')
    
    
    generator = BuildGenerator(cbn=10,resblock3=False,noise=True,
                           spectral_normalization=True,
                           out_channels=1,init_shape=(7,7,256),
                           in_shape=(128,))
    
    
    print('DCGAN_Discriminator')
    model = BuildDiscriminator(resnet=False)
    plot_model(model, show_shapes=True, to_file=DIR+'DCGAN_Discriminator.png')
    
    print('ResNet_Discriminator')
    #MNIST
    model = BuildDiscriminator(cbn=10,in_shape=(28,28,1))
    #CIFAR
    model = BuildDiscriminator(cbn=10,in_shape=(32,32,3))
    plot_model(model, show_shapes=True, to_file=DIR+'ResNet_Discriminator.png')
    
    
    print('Generator_resblock_1')
    model = ResBlock(input_shape=(4,4,256), sampling='up',  name='Generator_resblock_1')
    plot_model(model, show_shapes=True, to_file=DIR+'Generator_resblock_1.png')

    
    print('Discriminator_resblock_Down_1')
    model = ResBlock(input_shape=(32,32,3), channels=128, sampling='down', spectral_normalization=True, name='Discriminator_resblock_Down_1')
    plot_model(model, show_shapes=True, to_file=DIR+'Discriminator_resblock_Down_1.png')
    
    print('Discriminator_resblock_1')
    model = ResBlock(input_shape=(8,8,128),channels=128 , sampling=None, spectral_normalization=True, name='Discriminator_resblock_1' )
    plot_model(model, show_shapes=True, to_file=DIR+'Discriminator_resblock_1.png')