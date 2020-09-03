import sys
sys.path.append("..")
from utils.data import sample_known_unknown_classes

setup = {}

#Dataset Settings
ds_dict = {}
ds_dict['name'] = 'MNIST' #CIFAR10
ds_dict['image_shape'] = (28,28,1)
if 'CIFAR' in ds_dict['name']:
    ds_dict['image_shape'] = (32,32,3)
ds_dict['n_classes'] = 10
if ds_dict['name'] == 'CIFAR100':
    ds_dict['n_classes'] = 100

ds_dict['n_known'] = 6

#For random split
# _,ds_dict['known_classes'],\
#     ds_dict['unknown_classes'] = \
#         sample_known_unknown_classes(ds_dict['n_classes'],
#                                     ds_dict['n_known'])
ds_dict['known_classes'] = [9, 6, 0, 5, 3, 7]
ds_dict['unknown_classes'] = [8, 1, 2, 4]

ds_dict['known_class_mapping'] = {}
ds_dict['known_class_mapping_inv'] = {}
for i,kc in enumerate(ds_dict['known_classes']):
    ds_dict['known_class_mapping'][kc] = i
    ds_dict['known_class_mapping_inv'][i] = kc

setup['dataset'] = ds_dict

#Approach Settings
setup['mismatch_expected_recon'] = 'noise' #invert #image
setup['mismatch_expected_discriminator'] = 'fake' #image #mismatch #invert

#Training Settings - TODO Separate dict for stage1, stage2 training
train_dict = {}
#train_dict['batch_size'] = 64
#train_dict['epochs'] = 150
train_dict['training_ratio'] = 2
train_dict['sample_mismatch_every_epoch'] = True
setup['stage2_train'] = train_dict

#Adam Optimizer Settings
opt_dict = {}
opt_dict['beta_1'] = 0.
opt_dict['beta_2'] = 0.9
opt_dict['lr'] = 2e-4
setup['stage2_optimizer'] = opt_dict

#Debug Settings
debug_dict = {}
debug_dict['num_rows'] = 8
debug_dict['batch_size'] = debug_dict['num_rows']**2
debug_dict['dir'] = 'img/test/'
setup['debug'] = debug_dict

#Model Settings
model_dict = {}
#Encoder & Classifier Settings
model_dict['latent_dim'] = 128
model_dict['bn_momentum'] = 0.9
model_dict['bn_epsilon'] = 2e-5

#Generator Settings
generator_dict = {}
generator_dict['noise'] = False
generator_dict['SN'] = True #Spectral Normalization
generator_dict['out_channels'] = setup['dataset']['image_shape'][-1]
generator_dict['lambda'] = 0.95 #FM Loss Weight; CS Weight = 1-FM Weight
generator_dict['beta'] = 0.7 #Match Weight; Mismatch Weight = 1-Match Weight
generator_dict['gamma'] = 1. #Reconstruction Loss Weight
generator_dict['cbn'] = setup['dataset']['n_known']
generator_dict['init_shape'] = (7,7,256) #For MNIST ---TODO for remaining
generator_dict['in_shape'] = (model_dict['latent_dim'],)
generator_dict['resblock3'] = False

if not setup['dataset']['name'] == 'MNIST':
    generator_dict['resblock3'] = True
model_dict['generator'] = generator_dict

#Discriminator Settings
discriminator_dict = {}
if setup['mismatch_expected_discriminator'] == 'fake' or setup['mismatch_expected_discriminator'] == 'image':
    discriminator_dict['n_classes'] = setup['dataset']['n_known'] + 1 #N+1 Way Classification
elif setup['mismatch_expected_discriminator'] == 'mismatch':
    discriminator_dict['n_classes'] = setup['dataset']['n_known'] + 2 #N+2 Way Classification
elif setup['mismatch_expected_discriminator'] == 'invert':
    discriminator_dict['n_classes'] = 2*setup['dataset']['n_known'] + 1 #2N+1 Way Classification

model_dict['discriminator'] = discriminator_dict
setup['model'] = model_dict

#Checkpointing Settings
checkpoint_dict = {}
checkpoint_dict['classifier_save_dir'] = 'models/classifier/'+setup['dataset']['name']+'/'
checkpoint_dict['cGAN_save_dir'] = 'models/cGAN/'+setup['dataset']['name']+'/'
setup['checkpoint'] = checkpoint_dict

#Setup Approaches
experiment_parameters = {}
experiment_parameters['1a'] = setup
experiment_parameters['1b'] = setup
experiment_parameters['2'] = setup
experiment_parameters['3a'] = setup
experiment_parameters['3b'] = setup
experiment_parameters['4'] = setup

#Approach 1
experiment_parameters['1a']['mismatch_expected_recon'] = 'noise'
experiment_parameters['1a']['mismatch_expected_discriminator'] = 'fake'
experiment_parameters['1b']['mismatch_expected_recon'] = 'invert'
experiment_parameters['1b']['mismatch_expected_discriminator'] = 'fake'

#Approach 2
experiment_parameters['2']['mismatch_expected_recon'] = 'image'
experiment_parameters['2']['mismatch_expected_discriminator'] = 'image'

#Approach 3
experiment_parameters['3a']['mismatch_expected_recon'] = 'noise'
experiment_parameters['3a']['mismatch_expected_discriminator'] = 'mismatch'
experiment_parameters['3b']['mismatch_expected_recon'] = 'invert'
experiment_parameters['3b']['mismatch_expected_discriminator'] = 'mismatch'

#Approach 4
experiment_parameters['4']['mismatch_expected_recon'] = 'invert'
experiment_parameters['4']['mismatch_expected_discriminator'] = 'invert'