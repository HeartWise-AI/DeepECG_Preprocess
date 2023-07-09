import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['CUDA_VISIBLE_DEVICES']='0,1'
os.environ['WANDB_WATCH'] = 'false'
#os.environ["TF_GPU_ALLOCATOR"]="cuda_malloc_async"

import wandb
import gc
import pprint
from classification_models_1D.tfkeras import Classifiers
from termcolor import colored
import yaml
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import tensorflow as tf
import numpy as np
from statistics import mean, stdev 
import pickle
pp = pprint.PrettyPrinter(depth=4)
from classification_models_1D.tfkeras import Classifiers
import random
import sys  
import functools
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization, Activation, InputLayer, SeparableConv1D
from keras import Model
import tensorflow_addons as tfa
from tensorflow.keras.utils import Sequence
from keras import backend as K
from focal_loss import BinaryFocalLoss


tensorflow_version = float(tf.__version__[0:3])

# Get the GPU memory fraction to allocate
gpu_memory_fraction = 0.5
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
session = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

wandb.login()

from termcolor import colored

def step_color(bloc_order, position):
    formattedText = []
    for pos, i in enumerate(bloc_order):
        pos += 1
        if pos != position and pos != len(bloc_order):
            formattedText.append('{} -> '.format(i))
        elif pos == position:
            formattedText.append(colored('{}'.format(i),'white','on_red'))
            formattedText.append(' -> ')

        else:
            formattedText.append(i)

    return ''.join(formattedText)

def init_wandb(dict_yaml, step = 'model'):
    
    if step == 'model':
        config={
            'name': dict_yaml['name_sweep'] + '_models',
            'method': 'grid',
            'metric': {'goal': dict_yaml['metric']['goal'], 'name': dict_yaml['metric']['name']},
            'parameters':
            {
                'models': {'values':dict_yaml['config']['models']}

            }
        }    

    return config

def set_seeds(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)    

with open('test.yml', 'r') as file:
    prime_service = yaml.safe_load(file)

#print user paraeters

print(colored('Curently running core model optimization', 'green'))
print(colored('Job name :', 'green'), colored(prime_service['run_name'], 'cyan'))
print(colored('Sweep name :', 'green'), colored(prime_service['name_sweep'], 'cyan'))
print(colored('Optimization method :', 'green'), colored(prime_service['method'], 'cyan'))
print(colored('Optimization applied to the metric :', 'green'), colored(prime_service['metric']['name'], 'cyan'))
print(colored('Optimization goal :', 'green'), colored(prime_service['metric']['goal'], 'cyan'))
print('\n')
print(colored('Chosen parameters are:', 'green'))
pp.pprint(prime_service)

block_counter = 1
print('\n')
print(colored('Currently running:','green'))
print(step_color(prime_service['bloc_order'], block_counter))
print('\n')
print(colored('Step {}.1:'.format(block_counter),'green'))
print(colored('Main backbone'))

# select a model to run
sweep_configuration = init_wandb(prime_service)
print(sweep_configuration)
sweep_id = wandb.sweep(sweep=sweep_configuration, project=prime_service['run_name'])


from tensorflow.keras.utils import Sequence
import numpy as np   

class DataGenerator(Sequence):
    def __init__(self, x_set, y_set, batch_size):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        return batch_x, batch_y



def block_1_1(prime_service):

    run = wandb.init()

    save_dict = {}

    #load model
    print(wandb.config)
    loaded_architecture, _ = Classifiers.get(wandb.config.models)
    print('Currently running {}'.format(wandb.config.models))

    base = loaded_architecture(
    input_shape=(2500, 12),
    include_top=False,
    weights=None,
    pooling='max'
    )
    x = Flatten()(base.output)
    #print(x)
    #x = GlobalAveragePooling1D()(x)
    x = Dense(77, activation='sigmoid')(x)
    model = Model(inputs=base.inputs, outputs=x)

    optimizer_ = tf.keras.optimizers.Adam(learning_rate=0.01)
    loss_func = 'binary_crossentropy'

    model.compile(optimizer=optimizer_, loss=loss_func, metrics=['categorical_accuracy',
                                                                 'binary_accuracy', 
                                                                 tf.keras.metrics.AUC(curve='ROC',multi_label=True,num_labels=77),
                                                                 tf.keras.metrics.AUC(curve='PR',multi_label=True,num_labels=77, name='PR'),
                                                                 tfa.metrics.F1Score(num_classes=77,average='micro', name='F1_micro'), 
                                                                 tfa.metrics.F1Score(num_classes=77,average='macro', name='F1_macro'), 
                                                                 tf.keras.metrics.SpecificityAtSensitivity(0.5,name='specificity'), 
                                                                 tf.keras.metrics.SensitivityAtSpecificity(0.5,name='sensitivity')], run_eagerly=False)

    early_stop = EarlyStopping(monitor='val_loss', patience=5, verbose=2)
    lrplateau = ReduceLROnPlateau(  
    monitor="val_loss",
    factor=0.1,
    patience=3,
    verbose=0,
    mode="auto", 
    min_delta=0.0001,
    cooldown=0,
    min_lr=1e-12)

    #load data
    categorical_accuracy_list = list()
    binary_accuracy_list = list()
    AUC_list = list()
    PR_list = list()
    F1_micro_list = list()
    F1_macro_list = list()
    Sensitivity_list = list()
    Specificity_list = list()
    Hamming_list = list()

    list_of_metrics = [categorical_accuracy_list,binary_accuracy_list,AUC_list,PR_list,F1_micro_list,F1_macro_list,Sensitivity_list,Specificity_list,Hamming_list]

    try:
        os.mkdir(os.path.join(prime_service['save_dir'],wandb.config.models))
    except:
        pass

    for split in range(3):
        gc.collect()
        set_seeds(prime_service['three_seeds'][split])
        try:
            os.mkdir(os.path.join(prime_service['save_dir'],wandb.config.models,'split_{}'.format(split)))
        except:
            pass        

        X_train = np.load("/media/data1/anolin/new_split/split_{}/train_X.npy".format(split))
        Y_train = np.load("/media/data1/anolin/new_split/split_{}/train_Y.npy".format(split))

        X_val = np.load("/media/data1/anolin/new_split/split_{}/val_X.npy".format(split))
        Y_val = np.load("/media/data1/anolin/new_split/split_{}/val_Y.npy".format(split))

        train_gen = DataGenerator(X_train, Y_train, prime_service['batch_size'])
        val_gen = DataGenerator(X_val, Y_val,  prime_service['batch_size'])


        history = model.fit(train_gen,
            validation_data=val_gen,
            epochs=prime_service['num_epochs'],
            callbacks=[lrplateau,early_stop],
            verbose=1)

        #del var to make some space
        del X_train
        del Y_train
        del X_val
        del Y_val

        #evaluate model
        X_test = np.load("/media/data1/anolin/new_split/split_{}/test_X.npy".format(str(split)))
        Y_test = np.load("/media/data1/anolin/new_split/split_{}/test_Y.npy".format(str(split)))
        test_gen = DataGenerator(X_test, Y_test, prime_service['batch_size'])
        metrics = model.evaluate(test_gen)

        del X_test
        del Y_test

        dict_results = dict()
        for pos,i in enumerate(['categorical_accuracy','binary_accuracy','AUC','PR','F1_micro','F1_macro','Sensitivity','Specificity']):
            dict_results.update({i:metrics[pos]})
            list_of_metrics[pos].append(metrics[pos])

        #save out metrics
        save_dict.update({'{}_{}'.format(wandb.config.models,split):{'history':history.history, 'test_performances':dict_results}})

        #save model weights
        model.save_weights(os.path.join(prime_service['save_dir'],wandb.config.models,'split_{}'.format(split),'{}_{}.h5'.format(wandb.config.models,split)))
        gc.collect()


    print('Perfroances for {} were {} ± {}'.format(wandb.config.models,mean(list_of_metrics[3]), stdev(list_of_metrics[3])))

    with open(os.path.join(prime_service['save_dir'],wandb.config.models,'saved_outputs_detailed.pickle'), 'wb') as handle:
        pickle.dump(save_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    wandb.log({
    'model': wandb.config.models, 
    'categorical_accuracy': mean(list_of_metrics[0]),
    'binary_accuracy': mean(list_of_metrics[1]),
    'AUC': mean(list_of_metrics[2]),
    'PR': mean(list_of_metrics[3]),
    'F1_micro':mean(list_of_metrics[4]),
    'F1_macro':mean(list_of_metrics[5]),
    'Sensitivity': mean(list_of_metrics[6]),
    'Specificity': mean(list_of_metrics[7])})

wandb.agent(sweep_id, function=functools.partial(block_1_1,prime_service), count=len(prime_service['config']['models']))
wandb.finish()


