from tensorflow.python.keras.layers import Input, Dense, Conv2D, Add, Dot, Conv2DTranspose,Conv1D,\
 Activation, Reshape,BatchNormalization,UpSampling2D,AveragePooling2D, \
 GlobalAveragePooling2D, LeakyReLU, Flatten, Concatenate, Embedding
from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.keras import backend as K
from tensorflow.keras.utils import plot_model
from tensorflow.python.keras.layers.pooling import GlobalPooling2D
from utils.SN import SpectralNormalization
from utils.CBN import ConditionalAffine


inp = Input(shape=(10,2))
denseSN = SpectralNormalization(Dense(10))
x = denseSN(inp)
model = Model(inp,x)

model.summary()