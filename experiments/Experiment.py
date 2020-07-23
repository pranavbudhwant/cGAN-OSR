#Define the class Experiment, which contains the train methods, parameters, data
import sys
sys.path.append("..")

import tensorflow as tf

print("TF built with GPU support: ", tf.test.is_built_with_cuda())
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

from tensorflow.keras.datasets import mnist, cifar10, cifar100
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import joblib
import os

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

#Local modules
from config import experiment_parameters
from utils.data import onehotencode, get_mapped_labels
from model import BuildEncoder, BuildClassifier, GlobalSumPooling2D
from pyimagesearch.learningratefinder import LearningRateFinder
from pyimagesearch.clr_callback import CyclicLR


BASE_DIR = 'D:/dev/cGAN-OSR/'

def auroc(y_true, y_pred):
    try:
        return tf.py_function(roc_auc_score, (y_true, y_pred), tf.double)
    except:
        return -1

class Experiment:
    def __init__(self, experiment_params):
        self.params = experiment_params

        #Make checkpoint dirs
        if not os.path.exists(BASE_DIR+self.params['checkpoint']['classifier_save_dir']):
            os.makedirs(BASE_DIR+self.params['checkpoint']['classifier_save_dir'])
            print('Classifier checkpoint directory created')
        
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
        filepath = BASE_DIR+self.params['checkpoint']['classifier_save_dir']+'classifier-save-{epoch:02d}-{loss:.3f}-{val_loss:.3f}.hdf5'
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


if __name__ == "__main__":
    experiment = Experiment(experiment_parameters['1a'])
    experiment.load_classifier_data(summary=True)
    experiment.load_stage1_models(summary=False, file='classifier-save-01-5.666-0.000.hdf5')

    experiment.find_classifier_LR()

    nsamples = experiment.x_train.shape[0]
    batch_size = 128
    MIN_LR = 1e-6
    MAX_LR = 1e-4
    STEP_SIZE = 2*np.ceil((nsamples/float(batch_size)))
    CLR_METHOD = "triangular2"
    clr = CyclicLR(
        mode=CLR_METHOD,
        base_lr=MIN_LR,
        max_lr=MAX_LR,
        step_size=STEP_SIZE)

    experiment.train_stage1(MIN_LR,batch_size,5,clr)

    experiment.test_stage1()

    experiment.visualize_classifier_embeddings()
