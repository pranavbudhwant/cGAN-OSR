#Define the class Experiment, which contains the train methods, parameters, data
import sys
sys.path.append("..")

import tensorflow as tf
import tensorflow.keras.backend as K

print("TF built with GPU support: ", tf.test.is_built_with_cuda())
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

from tensorflow.keras.datasets import mnist, cifar10, cifar100
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Input
from tensorflow.keras.utils import Progbar

from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import joblib
import os
from time import time

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

#Local modules
from config import experiment_parameters
from utils.data import onehotencode,sample_known_unknown_classes,\
sample_mismatch_labels,sample_mismatch_images,get_classwise, get_mapped_labels,\
get_mismatch_data, get_known_unknown_data
from model import BuildEncoder, BuildClassifier, GlobalSumPooling2D, BuildGenerator, BuildDiscriminator, BuildDiscriminatorCS
from pyimagesearch.learningratefinder import LearningRateFinder
from pyimagesearch.clr_callback import CyclicLR


BASE_DIR = 'D:/dev/cGAN-OSR/'

def auroc(y_true, y_pred):
    try:
        return tf.py_function(roc_auc_score, (y_true, y_pred), tf.double)
    except:
        return -1

def wasserstein_loss(y_true, y_pred):
    return K.mean(y_true*y_pred)

def crammer_singer_criterion(y_true, y_pred):
    mask = tf.math.equal(y_true,1) #Get indices for target labels
    mask_inv = tf.logical_not(mask) #Get indices for other labels
    
    shape1 = tf.stack((tf.shape(y_pred)[0],))
    shape2 = tf.stack((tf.shape(y_pred)[0],tf.shape(y_pred)[1]-1))
    
    #Predicted values for target label
    target = tf.reshape(tf.boolean_mask(y_pred,mask),shape1) 
    
    #Max predicted value for wrong label
    max_wrong = tf.math.reduce_max(tf.reshape(tf.boolean_mask(y_pred,mask_inv),
                                              shape2),axis=1)
    
    return tf.reduce_mean(tf.nn.relu(1 + max_wrong - target))

def fm_loss(y_true, y_pred):
    target_features_mean = tf.reduce_mean(y_true,axis=0)
    pred_features_mean = tf.reduce_mean(y_pred,axis=0)
    abs_diff = tf.math.abs(target_features_mean - pred_features_mean)
    return tf.reduce_mean(abs_diff)

class Experiment:
    def __init__(self, experiment_params):
        self.params = experiment_params

        #Make classifier_save checkpoint dirs
        if not os.path.exists(BASE_DIR+self.params['checkpoint']['classifier_save_dir']):
            os.makedirs(BASE_DIR+self.params['checkpoint']['classifier_save_dir'])
            print('Classifier checkpoint directory created')

        #Make cGAN_save checkpoint dirs
        if not os.path.exists(BASE_DIR+self.params['checkpoint']['cGAN_save_dir']):
            os.makedirs(BASE_DIR+self.params['checkpoint']['cGAN_save_dir'])
            print('cGAN checkpoint directory created')

        #Make debug dir
        if not os.path.exists(BASE_DIR+self.params['debug']['dir']):
            os.makedirs(BASE_DIR+self.params['debug']['dir'])
            print('Debug directory creted')

    def load_classifier_data(self, summary=False):
        dataset_params = self.params['dataset']
        if dataset_params['name'] == 'MNIST':
            dataset = mnist
        elif dataset_params['name'] == 'CIFAR10':
            dataset = cifar10
        elif dataset_params['name'] == 'CIFAR100':
            dataset = cifar100
        else:
            raise Exception("Undefined Dataset in Experiment Params")

        #Load data
        (x_train, y_train), (x_test, y_test) = dataset.load_data()

        #Normalize
        x_train = (x_train - 127.5)/127.5
        x_test = (x_test - 127.5)/127.5

        #Configure training data
        y_train_known = y_train[np.where(np.isin(y_train, dataset_params['known_classes']))]
        #y_train_unknown = y_train[np.where(np.isin(y_train, dataset_params['unknown_classes']))]

        x_train_known = x_train[np.where(np.isin(y_train,dataset_params['known_classes']))]
        #x_train_unknown = x_train[np.where(np.isin(y_train,unknown_classes))]

        y_test_known = y_test[np.where(np.isin(y_test,dataset_params['known_classes']))]
        x_test_known = x_test[np.where(np.isin(y_test,dataset_params['known_classes']))]

        y_train_known_mapped = get_mapped_labels(y_train_known, dataset_params['known_class_mapping'])
        y_test_known_mapped = get_mapped_labels(y_test_known, dataset_params['known_class_mapping'])

        y_train_known_ohe = onehotencode(y_train_known_mapped, wgan=False)
        y_test_known_ohe = onehotencode(y_test_known_mapped, wgan=False)

        if len(x_train_known.shape)<4:
            x_train_known = np.expand_dims(x_train_known,axis=-1)
            x_test_known = np.expand_dims(x_test_known,axis=-1)
            
        del x_train, y_train, x_test, y_test

        x_train, x_val, y_train, y_val = train_test_split(x_train_known,y_train_known_ohe,
                                                        test_size=0.1,random_state=10)

        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.x_test = x_test_known
        self.y_test_ohe = y_test_known_ohe
        self.y_test_known_mapped = y_test_known_mapped

        if summary:
            print("Classifier Data Loaded")
            print("Shapes:")
            print("x_train:", self.x_train.shape)
            print("y_train:", self.y_train.shape)
            print("x_val  :", self.x_val.shape)
            print("y_val  :", self.y_val.shape)
            print("x_test :", self.x_test.shape)
            print("y_test :", self.y_test_ohe.shape)

    def load_stage1_models(self, summary=True, file=None):
        if file:
            self.classifier = load_model(BASE_DIR+self.params['checkpoint']['classifier_save_dir']+file,
                                custom_objects={'GlobalSumPooling2D':GlobalSumPooling2D,
                                'auroc':auroc})
            self.encoder = self.classifier.layers[1]
        else:
            #Stage 1 Models: Encoder, Optional Decoder, Classifier
            img_shape = self.params['dataset']['image_shape']
            latent_dim = self.params['model']['latent_dim']

            #Build encoder
            self.encoder = BuildEncoder(in_shape=img_shape, latent_dim=latent_dim,
                                    summary=summary)
            
            #Build classifier
            self.classifier = BuildClassifier(self.encoder, in_shape=img_shape, 
                            num_classes = len(self.params['dataset']['known_classes']),summary=summary)
        
        #TODO Build Optional Decoder

    def find_classifier_LR(self, batch_size=128, start_LR=1e-8, end_LR=1e-1):
        nsamples = self.x_train.shape[0]
        self.classifier.compile(optimizer=Adam(0.001),
                                loss='categorical_crossentropy')
        
        lrf = LearningRateFinder(self.classifier)
        
        lrf.find(
            trainData=[self.x_train,self.y_train],
            startLR=1e-8, endLR=1e-2,
            useGen=False,
            epochs=5,
            stepsPerEpoch=np.ceil((nsamples/float(batch_size))),
            batchSize=batch_size,
            verbose=1)

        lrf.plot_loss()

    def train_stage1(self, lr, batch_size, epochs, clr=None, early_stopping=False):
        #TODO Add training code for optional decoder
        
        #Configure checkpointing
        filepath = BASE_DIR+self.params['checkpoint']['classifier_save_dir']+\
            'classifier-save-{epoch:02d}-{loss:.3f}-{val_loss:.3f}.hdf5'
        checkpoint = ModelCheckpoint(filepath, monitor='val_loss',
                             save_weights_only=False,
                             verbose=1, save_best_only=False, mode='min',
                             period=1)
        
        #Callbacks
        earlystopping = EarlyStopping(monitor='val_loss',verbose=1,mode='min')
        callbacks_list = [checkpoint]
        if early_stopping:
            callbacks_list += [earlystopping]
        if clr is not None:
            callbacks_list += [clr]

        self.classifier.compile(optimizer=Adam(lr),loss='categorical_crossentropy',
                           metrics=['categorical_accuracy',auroc])

        self.classifier_train_history = self.classifier.fit(x=self.x_train, y=self.y_train,
                                    batch_size=batch_size,
                                    epochs=epochs,
                                    validation_data=[self.x_val, self.y_val],
                                    callbacks=callbacks_list,verbose=1)
        
        #Save train history
        joblib.dump(self.classifier_train_history.history, BASE_DIR+self.params['checkpoint']['classifier_save_dir']+'classifier_train_history_epoch_'+str(epochs)+'.pkl')

        #Plot loss
        #fig = 
        plt.figure()
        #plt.subplot(131)
        plt.plot(self.classifier_train_history.history['loss'],label='loss')
        plt.plot(self.classifier_train_history.history['val_loss'],label='val_loss')
        plt.legend(loc='best')
        plt.show()

        #Plot Accuracy
        #fig = 
        plt.figure()
        #plt.subplot(132)
        plt.plot(self.classifier_train_history.history['categorical_accuracy'],label='categorical_accuracy')
        plt.plot(self.classifier_train_history.history['val_categorical_accuracy'],label='val_categorical_accuracy')
        plt.legend(loc='best')
        plt.show()

        #Plot AUROC
        #fig = 
        plt.figure()
        #plt.subplot(133)
        plt.plot(self.classifier_train_history.history['auroc'],label='auroc')
        plt.plot(self.classifier_train_history.history['val_auroc'],label='val_auroc')
        plt.legend(loc='best')
        plt.show()

    def test_stage1(self):
        y_test_preds = self.classifier.predict(self.x_test)
        auc = roc_auc_score(self.y_test_known_mapped,y_test_preds,multi_class='ovo')
        acc = accuracy_score(self.y_test_known_mapped,np.argmax(y_test_preds,axis=1))

        print('Test Set Results:')
        print('ROC AUC Score : ', auc)
        print('Accuracy Score:', acc)

    def visualize_classifier_embeddings(self, tsne=False, threeD=False):
        encoder = self.classifier.layers[1]
        train_vecs = encoder.predict(self.x_train)
        #val_vecs = encoder.predict(self.x_val)

        if not tsne:
            pca = PCA(n_components=3)
            pca_result = pca.fit_transform(train_vecs)

            df = pd.DataFrame()
            df['pca-one'] = pca_result[:,0]
            df['pca-two'] = pca_result[:,1]
            df['pca-three'] = pca_result[:,2]
            df['y'] = np.argmax(self.y_train,axis=1)

            print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))

            if not threeD:
                plt.figure(figsize=(16,10))
                sns.scatterplot(
                    x="pca-one", y="pca-two",
                    hue="y",
                    palette=sns.color_palette("hls", len(self.params['dataset']['known_classes'])),
                    data=df,
                    legend="full",
                    alpha=0.3
                )
            else:
                ax = plt.figure(figsize=(16,10)).gca(projection='3d')
                ax.scatter(
                    xs=df["pca-one"], 
                    ys=df["pca-two"], 
                    zs=df["pca-three"], 
                    c=df["y"],
                    cmap='tab10'
                )
                ax.set_xlabel('pca-one')
                ax.set_ylabel('pca-two')
                ax.set_zlabel('pca-three')
                plt.show()
        else:
            tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
            tsne_results = tsne.fit_transform(train_vecs)

            if not threeD:
                df = pd.DataFrame()
                df['tsne-2d-one'] = tsne_results[:,0]
                df['tsne-2d-two'] = tsne_results[:,1]
                df['y'] = np.argmax(self.y_train,axis=1)

                plt.figure(figsize=(16,10))
                sns.scatterplot(
                    x="tsne-2d-one", y="tsne-2d-two",
                    hue="y",
                    palette=sns.color_palette("hls", len(self.params['dataset']['known_classes'])),
                    data=df,
                    legend="full",
                    alpha=0.3
                )
            else:
                #TODO tSNE 3D
                pass

    def load_cGAN_OSR_data(self , summary=False):
        dataset_params = self.params['dataset']
        if dataset_params['name'] == 'MNIST':
            dataset = mnist
        elif dataset_params['name'] == 'CIFAR10':
            dataset = cifar10
        elif dataset_params['name'] == 'CIFAR100':
            dataset = cifar100
        else:
            raise Exception("Undefined Dataset in Experiment Params")
            
        (x_train, y_train), (x_test, y_test) = dataset.load_data()

        x_train = (x_train - 127.5)/127.5
        x_test = (x_test - 127.5)/127.5

        assert (np.min(x_train) == -1) and (np.max(x_train) == 1)
        assert (np.min(x_test) == -1) and (np.max(x_test) == 1)

        (x_train_known,x_train_known_classwise,y_train_known,y_train_known_mapped,y_train_known_mapped_ohe),\
        (x_train_known_mismatch, y_train_known_mismatch, y_train_known_mismatch_mapped, y_train_known_mismatch_mapped_ohe),\
        (x_train_unknown) = get_known_unknown_data(x_train,y_train,dataset_params['known_classes'],dataset_params['known_class_mapping'],dataset_params['unknown_classes'],unknown=True)

        (x_test_known,x_test_known_classwise,y_test_known,y_test_known_mapped,y_test_known_mapped_ohe),\
        (x_test_known_mismatch, y_test_known_mismatch, y_test_known_mismatch_mapped, y_test_known_mismatch_mapped_ohe),\
        (x_test_unknown) = get_known_unknown_data(x_test,y_test,dataset_params['known_classes'],dataset_params['known_class_mapping'],dataset_params['unknown_classes'],unknown=True)

        self.x_train_known = x_train_known
        self.x_train_known_classwise = x_train_known_classwise
        self.y_train_known = y_train_known
        self.y_train_known_mapped = y_train_known_mapped
        self.y_train_known_mapped_ohe = y_train_known_mapped_ohe
        self.x_train_known_mismatch = x_train_known_mismatch
        self.y_train_known_mismatch = y_train_known_mismatch
        self.y_train_known_mismatch_mapped = y_train_known_mismatch_mapped
        self.y_train_known_mismatch_mapped_ohe = y_train_known_mismatch_mapped_ohe
        self.x_train_unknown = x_train_unknown

        self.x_test_known = x_test_known
        self.x_test_known_classwise = x_test_known_classwise
        self.y_test_known = y_test_known
        self.y_test_known_mapped = y_test_known_mapped
        self.y_test_known_mapped_ohe = y_test_known_mapped_ohe
        self.x_test_known_mismatch = x_test_known_mismatch
        self.y_test_known_mismatch = y_test_known_mismatch
        self.y_test_known_mismatch_mapped = y_test_known_mismatch_mapped
        self.y_test_known_mismatch_mapped_ohe = y_test_known_mismatch_mapped_ohe
        self.x_test_unknown = x_test_unknown

    def build_model_for_training_generator(self, summary=False):
        Image_input_for_encoder = Input(shape=self.params['dataset']['image_shape'])
        Encoder_output = self.encoder(Image_input_for_encoder)
        Noise_input_for_training_generator = Input(shape=self.params['model']['generator']['in_shape'])

        #Merged_inputs = Concatenate()([Encoder_output,
        #                              Noise_input_for_training_generator])

        opt_params = self.params['stage2_optimizer']

        Match_class_input_for_training_generator = Input(shape=(1,),dtype='int32')
        Mismatch_class_input_for_training_generator = Input(shape=(1,),dtype='int32')

        Match_Generated_image = self.generator([Encoder_output,
                                                Noise_input_for_training_generator,
                                                Match_class_input_for_training_generator])
        Mismatch_Generated_image = self.generator([Encoder_output,
                                                    Noise_input_for_training_generator,
                                                    Mismatch_class_input_for_training_generator])

        Match_discriminator_output = self.discriminator(Match_Generated_image)
        Match_discriminator_feat_output = self.discriminator_feat(Match_Generated_image)

        Mismatch_discriminator_output = self.discriminator(Mismatch_Generated_image)
        Mismatch_discriminator_feat_output = self.discriminator_feat(Mismatch_Generated_image)

        self.model_for_training_generator = Model([Image_input_for_encoder,
                                                    Noise_input_for_training_generator,
                                                    Match_class_input_for_training_generator,
                                                    Mismatch_class_input_for_training_generator], 
                                                    [Match_discriminator_output, Match_discriminator_feat_output,
                                                    Mismatch_discriminator_output, Mismatch_discriminator_feat_output])

        self.generator.trainable=True
        self.encoder.trainable=False
        self.discriminator.trainable = False
        self.discriminator_feat.trainable = False

        lmbda = self.params['model']['generator']['lambda'] #FM Loss Weight
        beta = self.params['model']['generator']['beta'] #Match weight
        #Upscale the loss
        #lmbda *= 10
        #beta *= 10
        self.model_for_training_generator.compile(optimizer=Adam(opt_params['lr'], 
                                                            beta_1=opt_params['beta_1'], 
                                                            beta_2=opt_params['beta_2']), 
                                                    loss=[crammer_singer_criterion,fm_loss,
                                                        crammer_singer_criterion,fm_loss],
                                                    loss_weights=[beta*(1-lmbda),
                                                                beta*lmbda,
                                                                (1-beta)*(1-lmbda),
                                                                (1-beta)*lmbda])
        
        if summary:
            print("model_for_training_generator")
            self.model_for_training_generator.summary()

    def build_model_for_training_discriminator(self, summary=False):
        #WLOSS
        Real_image = Input(shape=self.params['dataset']['image_shape'])
        Mismatch_Real_image = Input(shape=self.params['dataset']['image_shape'])

        Match_class_input_for_training_generator = Input(shape=(1,),dtype='int32')
        Mismatch_class_input_for_training_generator = Input(shape=(1,),dtype='int32')

        Encoder_output = self.encoder(Real_image)

        Noise_input_for_training_discriminator = Input(shape=self.params['model']['generator']['in_shape'])
        #Merged_inputs = Concatenate()([Encoder_output,
        #                              Noise_input_for_training_discriminator])

        Match_Fake_image = self.generator([Encoder_output,
                                            Noise_input_for_training_discriminator,
                                            Match_class_input_for_training_generator])
        Mismatch_Fake_image = self.generator([Encoder_output,
                                                Noise_input_for_training_discriminator,
                                                Mismatch_class_input_for_training_generator])

        Discriminator_output_for_real = self.discriminator(Real_image)
        Discriminator_output_for_mismatch_real = self.discriminator(Mismatch_Real_image)

        Discriminator_output_for_match_fake = self.discriminator(Match_Fake_image)
        Discriminator_output_for_mismatch_fake = self.discriminator(Mismatch_Fake_image)

        self.model_for_training_discriminator = Model([Real_image,
                                                        Mismatch_Real_image,
                                                        Noise_input_for_training_discriminator,
                                                        Match_class_input_for_training_generator,
                                                        Mismatch_class_input_for_training_generator],
                                                        [Discriminator_output_for_real,
                                                        Discriminator_output_for_mismatch_real,
                                                        Discriminator_output_for_match_fake,
                                                        Discriminator_output_for_mismatch_fake])

        self.encoder.trainable = False
        self.generator.trainable = False
        self.discriminator_feat.trainable=True
        self.discriminator.trainable = True
        self.model_for_training_discriminator.compile(optimizer=Adam(self.params['stage2_optimizer']['lr'], 
                                                                beta_1=self.params['stage2_optimizer']['beta_1'], 
                                                                beta_2=self.params['stage2_optimizer']['beta_2']), 
                                                loss=[crammer_singer_criterion,
                                                    crammer_singer_criterion, 
                                                    crammer_singer_criterion,
                                                    crammer_singer_criterion])

        if summary:
            print("model_for_training_discriminator")
            self.model_for_training_discriminator.summary()

    def load_cGAN_OSR_models(self, classifier_file, summary=False):
        
        self.load_stage1_models(file=classifier_file)

        #Build Generator
        g_params = self.params['model_dict']['generator']
        self.generator = BuildGenerator(cbn=g_params['cbn'],
            noise = g_params['noise'],
            resblock3=g_params['resblock3'],
            spectral_normalization=g_params['SN'],
            out_channels=g_params['out_channels'],
            init_shape=g_params['init_shape'],
            in_shape=g_params['in_shape'],
            summary=summary)

        #Build Discriminators
        self.discriminator, self.discriminator_feat = BuildDiscriminatorCS(num_classes=len(self.params['dataset']['known_classes']),
                                                                            in_shape=self.params['dataset']['image_shape'],
                                                                            feat=True,
                                                                            summary=summary)

        #Build model for training Generator
        self.build_model_for_training_generator(summary=summary)

        #Build Model for training Discriminators
        self.build_model_for_training_discriminator(summary=summary)

    def train_stage_2(self , batch_size , epochs):
        debug_params = self.params['debug']
        train_params = self.params['stage2_train']

        #0th index is considered as fake in loss formulation, 
        fake_y = np.ones_like(self.y_train_known_mismatch_mapped_ohe)*-1
        fake_y[:,0] = 1

        print(self.y_train_known_mismatch_mapped[:5])
        print(self.y_train_known_mismatch_mapped_ohe[:5])
        print(fake_y[:5])

        assert np.sum(self.y_train_known_mismatch_mapped_ohe[:,0])*-1 == len(self.y_train_known_mismatch_mapped_ohe)
        assert np.sum(fake_y[:,0]) == len(fake_y)

        
        #Set the test data
        test_noise = np.random.randn(int(debug_params['batch_size']/2), self.params['model']['latent_dim'])

        test_class = self.y_train_known_mapped[:int(debug_params['batch_size']/2)]
        test_class_ohe = self.y_train_known_mapped_ohe[:int(debug_params['batch_size']/2)]
        test_images = self.x_train_known[:int(debug_params['batch_size']/2)]
        test_image_embeddings = self.encoder.predict(test_images)

        test_mismatch_class = self.y_train_known_mismatch_mapped[:int(debug_params['batch_size']/2)]
        test_mismatch_class_ohe = self.y_train_known_mismatch_mapped_ohe[:int(debug_params['batch_size']/2)]
        test_mismatch_images = self.x_train_known_mismatch[:int(debug_params['batch_size']/2)]
        test_mismatch_image_embeddings = self.encoder.predict(test_mismatch_images)

        test_mismatch_noise = np.random.randint(0,255,test_images.shape).astype(test_images.dtype)
        test_mismatch_noise = (test_mismatch_noise - 127.5)/127.5

        test_fake_ohe = fake_y[:int(debug_params['batch_size']/2)]

        print(test_class)

        image_dim = 28 #32
        image_channels = 1 #3

        CS_loss = []
        discriminator_loss = []
        generator_loss = []
        for epoch in range(epochs):
            x_train_known, y_train_known,\
            y_train_known_mapped, y_train_known_mapped_ohe = shuffle(self.x_train_known, 
                                                                    self.y_train_known,
                                                                    self.y_train_known_mapped, 
                                                                    self.y_train_known_mapped_ohe)

            x_train_known_mismatch, y_train_known_mismatch,\
            y_train_known_mismatch_mapped, y_train_known_mismatch_mapped_ohe = shuffle(self.x_train_known_mismatch, 
                                                                                    self.y_train_known_mismatch, 
                                                                                    self.y_train_known_mismatch_mapped, 
                                                                                    self.y_train_known_mismatch_mapped_ohe)

            #X,Y,real_y = shuffle(X,Y,real_y)
            
            print("epoch {} of {}".format(epoch+1, epochs))
            num_batches = int(x_train_known.shape[0] // batch_size)
            
            print("number of batches: {}".format(int(x_train_known.shape[0] // (batch_size))))
            
            progress_bar = Progbar(target=int(x_train_known.shape[0] // (batch_size * train_params['training_ratio'])))
            minibatches_size = batch_size * train_params['training_ratio']
            
            start_time = time()
            for index in range(int(x_train_known.shape[0] // (batch_size * train_params['training_ratio']))):
                progress_bar.update(index)
                
                minibatches_X = x_train_known[index * minibatches_size:(index + 1) * minibatches_size]
                minibatches_Y = y_train_known_mapped[index * minibatches_size:(index + 1) * minibatches_size]
                minibatches_Y_ohe = y_train_known_mapped_ohe[index * minibatches_size:(index + 1) * minibatches_size]
                
                minibatches_mismatch_X = x_train_known_mismatch[index * minibatches_size:(index + 1) * minibatches_size]
                minibatches_mismatch_Y = y_train_known_mismatch_mapped[index * minibatches_size:(index + 1) * minibatches_size]
                minibatches_mismatch_Y_ohe = y_train_known_mismatch_mapped_ohe[index * minibatches_size:(index + 1) * minibatches_size]
                
                #minibatches_real_y = real_y[index * minibatches_size:(index + 1) * minibatches_size]
                minibatches_fake_y = fake_y[index * minibatches_size:(index + 1) * minibatches_size]
                
                for j in range(train_params['training_ratio']):
                    image_batch = minibatches_X[j * batch_size : (j + 1) * batch_size]
                    match_class_batch = minibatches_Y[j * batch_size : (j + 1) * batch_size]
                    match_class_batch_ohe = minibatches_Y_ohe[j * batch_size : (j + 1) * batch_size]
                    
                    mismatch_image_batch = minibatches_mismatch_X[j * batch_size : (j + 1) * batch_size]
                    mismatch_class_batch = minibatches_mismatch_Y[j * batch_size : (j + 1) * batch_size]
                    mismatch_class_batch_ohe = minibatches_mismatch_Y_ohe[j * batch_size : (j + 1) * batch_size]
                    
                    #real_y_batch = minibatches_real_y[j * BATCHSIZE : (j + 1) * BATCHSIZE]
                    
                    fake_y_batch = minibatches_fake_y[j * batch_size : (j + 1) * batch_size]
                    
                    noise = np.random.randn(batch_size, 128).astype(np.float32)
                    
                    self.discriminator_feat.trainable = True
                    self.discriminator.trainable = True
                    self.generator.trainable = False
                    
                    discriminator_loss.append(self.model_for_training_discriminator.train_on_batch([image_batch,
                                                                                            mismatch_image_batch,
                                                                                            noise,
                                                                                            match_class_batch,
                                                                                            mismatch_class_batch],
                                                                                            [match_class_batch_ohe,
                                                                                            mismatch_class_batch_ohe,
                                                                                            fake_y_batch,
                                                                                            fake_y_batch]))
                self.discriminator_feat.trainable = False
                self.discriminator.trainable = False
                self.generator.trainable = True
                
                match_feat_batch = self.discriminator_feat.predict_on_batch(image_batch)
                mismatch_noise = np.random.randint(0,255,mismatch_image_batch.shape).astype(mismatch_image_batch.dtype)
                mismatch_noise = (mismatch_noise - 127.5)/127.5
                mismatch_feat_batch = self.discriminator_feat.predict_on_batch(mismatch_noise)
                #mismatch_feat_batch = discriminator_feat.predict_on_batch(mismatch_image_batch)
                
                generator_loss.append(self.model_for_training_generator.train_on_batch([image_batch,
                                                                                np.random.randn(batch_size, 128),
                                                                                match_class_batch,
                                                                                mismatch_class_batch], 
                                                                                [match_class_batch_ohe, 
                                                                                match_feat_batch, 
                                                                                fake_y_batch, #mismatch_class_batch_ohe 
                                                                                mismatch_feat_batch]))
            
            print('\nepoch time: {}'.format(time()-start_time))
                
            test_feat_batch = self.discriminator_feat.predict_on_batch(test_images)
            #test_mismatch_feat_batch = discriminator_feat.predict_on_batch(test_mismatch_images)
            test_mismatch_feat_batch = self.discriminator_feat.predict_on_batch(test_mismatch_noise)
            
            W_real = self.model_for_training_generator.evaluate([test_images,
                                                            test_noise,
                                                            test_class,
                                                            test_mismatch_class], 
                                                        [test_class_ohe,
                                                            test_feat_batch,
                                                            test_fake_ohe, #test_mismatch_class_ohe
                                                            test_mismatch_feat_batch])
            print(W_real)
            W_fake = self.model_for_training_generator.evaluate([test_images,
                                                            test_noise,
                                                            test_class,
                                                            test_mismatch_class], 
                                                        [test_fake_ohe,
                                                            test_feat_batch,
                                                            test_fake_ohe,
                                                            test_mismatch_feat_batch])
            print(W_fake)
            W_l = W_real+W_fake
            print('CS_loss: {}'.format(W_l))
            CS_loss.append(W_l)
            
            #Generate image
            generated_match_image = self.generator.predict([test_image_embeddings,
                                                    test_noise,
                                                    test_class])
            generated_mismatch_image = self.generator.predict([test_image_embeddings,
                                                        test_noise,
                                                        test_mismatch_class])
            
            generated_match_image = (generated_match_image+1)/2
            generated_mismatch_image = (generated_mismatch_image+1)/2
            
            generated_image = np.append(generated_match_image,generated_mismatch_image,axis=0)
            
            assert len(generated_image) == debug_params['batch_size']
            
            for i in range(debug_params['num_rows']):
                if image_channels == 1:
                    new = generated_image[i*debug_params['num_rows']:i*debug_params['num_rows']+
                                        debug_params['num_rows']].reshape(image_dim*debug_params['num_rows'],
                                        image_dim)
                else:
                    new = generated_image[i*debug_params['num_rows']:i*debug_params['num_rows']+
                                        debug_params['num_rows']].reshape(image_dim*debug_params['num_rows'],
                                        image_dim,image_channels)
                if i!=0:
                    old = np.concatenate((old,new),axis=1)
                else:
                    old = new
            print('plot generated_image')
            
            if image_channels == 1:
                plt.imsave('{}/SN_epoch_{}.png'.format(debug_params['dir'], epoch), old, cmap='gray')
            else:
                plt.imsave('{}/SN_epoch_{}.png'.format(debug_params['dir'], epoch), old)

    #For training cGAN w/ different loss formulations. Thus does not refer to Stage 2
    def load_cGAN_data(self, percent=1, summary=True):
        dataset_params = self.params['dataset']
        if dataset_params['name'] == 'MNIST':
            dataset = mnist
        elif dataset_params['name'] == 'CIFAR10':
            dataset = cifar10
        elif dataset_params['name'] == 'CIFAR100':
            dataset = cifar100
        else:
            raise Exception("Undefined Dataset in Experiment Params")

        #Load data
        (x_train, y_train), (x_test, y_test) = dataset.load_data()

        #Normalize
        x_train = (x_train - 127.5)/127.5
        x_test = (x_test - 127.5)/127.5

        X = np.concatenate((x_test,x_train))
        Y = np.concatenate((y_test,y_train))

        X, Y = shuffle(X, Y, random_state=10)
        
        #Keep percent% of data
        X = X[:int(len(X)*percent)]
        Y = Y[:int(len(Y)*percent)]

        if len(X.shape) < 4:
            X = np.expand_dims(X, axis=-1)

        self.X = X
        self.Y = Y

        if summary:
            print('cGAN Data Loaded')
            print('Shapes')
            print('X: ', X.shape)
            print('Y: ', Y.shape)

    def load_cGAN_models(self, epoch=None, summary=False):
        if epoch:
            cGAN_save_dir = BASE_DIR+self.params['checkpoint']['cGAN_save_dir']+'/'
            self.model_for_training_generator = load_model(cGAN_save_dir+\
                'model_for_training_generator_epoch_{:d}.h5'.format(epoch))
            self.model_for_training_discriminator = load_model(cGAN_save_dir+\
                'model_for_training_discriminator_epoch_{:d}.h5'.format(epoch))
            self.generator = load_model(cGAN_save_dir+\
                'generator_epoch_{:d}.h5'.format(epoch))
            self.discriminator = load_model(cGAN_save_dir+\
                'discriminator_epoch_{:d}.h5'.format(epoch))
            if summary:
                print("Generator")
                self.generator.summary()
                print("Discriminator")
                self.discriminator.summary()
                print("model_for_training_generator")
                self.model_for_training_generator.summary()
                print("model_for_training_discriminator")
                self.model_for_training_discriminator.summary()

            return


        g_params = self.params['model']['generator']
        ds_dict = self.params['dataset']

        #For WGAN, cbn = number of classes
        self.generator = BuildGenerator(cbn=ds_dict['n_classes'],
            noise = g_params['noise'],
            resblock3=g_params['resblock3'],
            spectral_normalization=g_params['SN'],
            out_channels=g_params['out_channels'],
            init_shape=g_params['init_shape'],
            in_shape=g_params['in_shape'],
            summary=summary)

        #For WGAN, embedding = number of classes
        self.discriminator = BuildDiscriminator(embedding=ds_dict['n_classes'],
            in_shape=self.params['dataset']['image_shape'],
            summary=summary)

        opt_params = self.params['stage2_optimizer']
        #For training generator
        Noise_input_for_training_generator = Input(shape=g_params['in_shape'])
        Class_input_for_training_generator = Input(shape=(1,),dtype='int32')
        Generated_image = self.generator([Noise_input_for_training_generator,
                                        Class_input_for_training_generator])
        Discriminator_output = self.discriminator([Generated_image,
                                                Class_input_for_training_generator])

        self.model_for_training_generator = Model([Noise_input_for_training_generator,
                                            Class_input_for_training_generator], 
                                            Discriminator_output)    
        self.discriminator.trainable = False
        self.model_for_training_generator.compile(optimizer=Adam(opt_params['lr'], 
                                                    beta_1=opt_params['beta_1'], 
                                                    beta_2=opt_params['beta_2']), 
                                                loss=wasserstein_loss)
        
        if summary:
            print("model_for_training_generator")
            self.model_for_training_generator.summary()

        #For training discriminator
        Real_image = Input(shape=self.params['dataset']['image_shape'])
        Noise_input_for_training_discriminator = Input(shape=g_params['in_shape'])
        Class_input_for_training_generator = Input(shape=(1,),dtype='int32')
        Fake_image = self.generator([Noise_input_for_training_discriminator,
                                    Class_input_for_training_generator])

        Discriminator_output_for_real = self.discriminator([Real_image,
                                                        Class_input_for_training_generator])
        Discriminator_output_for_fake = self.discriminator([Fake_image,
                                                        Class_input_for_training_generator])

        self.model_for_training_discriminator = Model([Real_image,
                                                Noise_input_for_training_discriminator,
                                                Class_input_for_training_generator],
                                                [Discriminator_output_for_real,
                                                Discriminator_output_for_fake])

        self.generator.trainable = False
        self.discriminator.trainable = True
        self.model_for_training_discriminator.compile(optimizer=Adam(opt_params['lr'], 
                                                        beta_1=opt_params['beta_1'], 
                                                        beta_2=opt_params['beta_2']), 
                                                loss=[wasserstein_loss, wasserstein_loss])

        if summary:
            print("model_for_training_discriminator")
            self.model_for_training_discriminator.summary()

    def train_cGAN(self, batch_size, epochs):
        debug_params = self.params['debug']
        cGAN_save_dir = BASE_DIR+self.params['checkpoint']['cGAN_save_dir']+'/'

        real_y = np.ones((batch_size, 1), dtype=np.float32)
        fake_y = -real_y

        test_noise = np.random.randn(debug_params['batch_size'], self.params['model']['latent_dim'])
        test_class = self.Y[:debug_params['batch_size']]
        test_real_y = np.ones((debug_params['batch_size'], 1), dtype=np.float32)
        test_fake_y = -test_real_y
        
        print('Test Classes: ', test_class)

        image_dim = self.params['dataset']['image_shape'][0]
        image_channels = self.params['dataset']['image_shape'][-1]
        
        W_loss = []
        discriminator_loss = []
        generator_loss = []
        
        for epoch in range(epochs):
            self.X,self.Y = shuffle(self.X,self.Y)
            
            print("epoch {} of {}".format(epoch+1, epochs))
            #num_batches = int(self.X.shape[0] // batch_size)
            
            print("number of batches: {}".format(int(self.X.shape[0] // (batch_size))))
            
            progress_bar = Progbar(target=int(self.X.shape[0] // (batch_size * self.params['stage2_train']['training_ratio'])))
            minibatches_size = batch_size * self.params['stage2_train']['training_ratio']
            
            start_time = time()
            for index in range(int(self.X.shape[0] // (batch_size * self.params['stage2_train']['training_ratio']))):
                progress_bar.update(index)
                discriminator_minibatches_X = self.X[index * minibatches_size:(index + 1) * minibatches_size]
                discriminator_minibatches_Y = self.Y[index * minibatches_size:(index + 1) * minibatches_size]
                
                for j in range(self.params['stage2_train']['training_ratio']):
                    image_batch = discriminator_minibatches_X[j * batch_size : (j + 1) * batch_size]
                    class_batch = discriminator_minibatches_Y[j * batch_size : (j + 1) * batch_size]
                    noise = np.random.randn(batch_size, 128).astype(np.float32)
                    self.discriminator.trainable = True
                    self.generator.trainable = False
                    discriminator_loss.append(self.model_for_training_discriminator.train_on_batch([image_batch, 
                                                                                            noise, 
                                                                                            class_batch],
                                                                                            [real_y, fake_y]))
                self.discriminator.trainable = False
                self.generator.trainable = True
                generator_loss.append(self.model_for_training_generator.train_on_batch([np.random.randn(batch_size, 128),
                                                                                class_batch], real_y))
                
            self.model_for_training_generator.save(cGAN_save_dir + \
                'model_for_training_generator_epoch_{:d}.h5'.format(epoch))
            self.model_for_training_discriminator.save(cGAN_save_dir + \
                'model_for_training_discriminator_epoch_{:d}.h5'.format(epoch))
            self.generator.save(cGAN_save_dir + 'generator_epoch_{:d}.h5'.format(epoch))
            self.discriminator.save(cGAN_save_dir + 'discriminator_epoch_{:d}.h5'.format(epoch))

            print('Models saved @ '+cGAN_save_dir)
            
            print('\nepoch time: {}'.format(time()-start_time))
            
            W_real = self.model_for_training_generator.evaluate([test_noise,
                                                            test_class], test_real_y)
            print(W_real)
            W_fake = self.model_for_training_generator.evaluate([test_noise,
                                                            test_class], test_fake_y)
            print(W_fake)
            W_l = W_real+W_fake
            print('wasserstein_loss: {}'.format(W_l))
            W_loss.append(W_l)
            #Generate image
            generated_image = self.generator.predict([test_noise,
                                                test_class])
            generated_image = (generated_image+1)/2
            for i in range(debug_params['num_rows']):
                if image_channels == 1:
                    new = generated_image[i*debug_params['num_rows']:\
                                          i*debug_params['num_rows']+\
                                              debug_params['num_rows']].reshape(\
                                              image_dim*debug_params['num_rows'],image_dim)
                else:
                    new = generated_image[i*debug_params['num_rows']:\
                                          i*debug_params['num_rows']+\
                                        debug_params['num_rows']].reshape(\
                                            image_dim*debug_params['num_rows'],image_dim,image_channels)
                if i!=0:
                    old = np.concatenate((old,new),axis=1)
                else:
                    old = new
            print('plot generated_image')
            
            if image_channels == 1:
                plt.imsave('{}cGAN_epoch_{}.png'.format(BASE_DIR+debug_params['dir'], epoch), old, cmap='gray')
            else:
                plt.imsave('{}cGAN_epoch_{}.png'.format(BASE_DIR+debug_params['dir'], epoch), old)


if __name__ == "__main__":

    #Debug Classifier Methods
    # experiment = Experiment(experiment_parameters['1a'])
    # experiment.load_classifier_data(summary=True)
    # experiment.load_stage1_models(summary=False, file='classifier-save-01-5.666-0.000.hdf5')

    # experiment.find_classifier_LR()

    # nsamples = experiment.x_train.shape[0]
    # batch_size = 128
    # MIN_LR = 1e-6
    # MAX_LR = 1e-4
    # STEP_SIZE = 2*np.ceil((nsamples/float(batch_size)))
    # CLR_METHOD = "triangular2"
    # clr = CyclicLR(
    #     mode=CLR_METHOD,
    #     base_lr=MIN_LR,
    #     max_lr=MAX_LR,
    #     step_size=STEP_SIZE)

    # experiment.train_stage1(MIN_LR,batch_size,5,clr)
    # experiment.test_stage1()
    # experiment.visualize_classifier_embeddings()

    #Debug cGAN methods
    # experiment = Experiment(experiment_parameters['1a'])
    # experiment.load_cGAN_data(percent=0.001, summary=True)
    # experiment.load_cGAN_models(summary=False)
    # experiment.train_cGAN(batch_size=8, epochs=1)

    #Debug cGAN-OSR methods
    pass
