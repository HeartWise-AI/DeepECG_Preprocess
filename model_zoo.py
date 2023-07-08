import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
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


tensorflow_version = float(tf.__version__[0:3])

# Get the GPU memory fraction to allocate
gpu_memory_fraction = 0.9

# Create GPUOptions with the fraction of GPU memory to allocate
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)

# Create a session with the GPUOptions
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

def init_wandb(dict_yaml):
    
    config={
        'name': dict_yaml['name_sweep'],
        'method': dict_yaml['method'],
        'metric': {'goal': dict_yaml['metric']['goal'], 'name': dict_yaml['metric']['name']},
        'parameters':
        {
            'models': {'values':dict_yaml['config']['models']}

        }
    }    

    return config

from keras import backend as K
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score

def macro_auc(y_true, y_pred):
    return roc_auc_score(y_true, y_pred, multi_class='ovr',average='macro')

def micro_auc(y_true, y_pred):
    return roc_auc_score(y_true, y_pred, multi_class='ovr', average='micro')

def sample_auc(y_true, y_pred):
    return roc_auc_score(y_true, y_pred,  multi_class='ovr' , average='samples')

def averaged_metric(y_true, y_pred):
    return (micro_auc(y_true, y_pred) + macro_auc(y_true, y_pred))/2


def auc_pr(y_true, y_pred):
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    return roc_auc_score(recall, precision)

def set_seeds(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)    

with open('test.yml', 'r') as file:
    prime_service = yaml.safe_load(file)
prime_service



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

data_dir = '/media/data1/anolin/new_split/'

def main(prime_service):
    #select vanilla model

    run = wandb.init(project=prime_service['run_name'])

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
                                                                 tf.keras.metrics.SensitivityAtSpecificity(0.5,name='sensitivity'),
                                                                 tfa.metrics.HammingLoss(mode='multilabel')], run_eagerly=False)

    early_stop = EarlyStopping(monitor='val_loss', patience=3, verbose=2)
    lrplateau = ReduceLROnPlateau(  
    monitor="val_loss",
    factor=0.1,
    patience=2,
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

    for split in range(3):
        gc.collect()

        set_seeds(prime_service['three_seeds'][split])

        X_train = np.load("/media/data1/anolin/split_smaller_for_ram/split_0/train_X_6k.npy".format(split))
        print(X_train.shape)
        Y_train = np.load("/media/data1/anolin/split_smaller_for_ram/split_0/train_Y_6k.npy".format(split))
        print(Y_train.shape)

        X_val = np.load("/media/data1/anolin/split_smaller_for_ram/split_0/val_X.npy".format(split))
        print(X_val.shape)
        Y_val = np.load("/media/data1/anolin/split_smaller_for_ram/split_0/val_Y.npy".format(split))


        model.fit(X_train, Y_train,
            validation_data=(X_val,Y_val),
            batch_size=16,
            use_multiprocessing=False,
            epochs=100,
            verbose=1)

        #evaluate model
        X_test = np.load("/media/data1/anolin/split_smaller_for_ram/split_0/test_X.npy".format(str(split)))
        Y_test = np.load("/media/data1/anolin/split_smaller_for_ram/split_0/test_Y.npy".format(str(split)))
        metrics = model.evaluate(x_test, y_test,batch_size=512)

        dict_results = dict()
        for pos,i in enumerate(['categorical_accuracy','binary_accuracy','AUC','PR','F1_micro','F1_macro','Sensitivity','Specificity','Hamming']):
            dict_results.update({i:metrics[pos]})
            list_of_metrics[pos].append(metrics[pos])


        if prime_service['save_info'] == True:
            #save out metrics
            save_dict.update({'{}_{}'.format(wandb.config.models,split):{'history':history.history, 'test_performances':dict_results, 'y_pred':Y_test_pred}})

            #save model weights
            model_selected.save_weights('{}_{}.h5'.format(wandb.config.models,split))
        
        gc.collect()


    print('Perfroances for {} were {} ± {}'.format(wandb.config.models,mean(list_scores), stdev(list_scores)))
    save_dict.update({wandb.config.models:mean(list_scores)})


    wandb.log({
    'model': wandb.config.models, 
    'categorical_accuracy': mean(list_of_metrics[0]),
    'binary_accuracy': mean(list_of_metrics[1]),
    'AUC': mean(list_of_metrics[2]),
    'PR': mean(list_of_metrics[3]),
    'F1_micro':mean(list_of_metrics[4]),
    'F1_macro':mean(list_of_metrics[5]),
    'Sensitivity': mean(list_of_metrics[6]),
    'Specificity': mean(list_of_metrics[7]),
    'Hamming': mean(list_of_metrics[8])
    })

    if prime_service['bloc_order'] == True:
        with open(os.path.join(prime_service['save_dir'],'saved_outputs_detailed.{}.step_{}.pickle'.format(prime_service['name_sweep'],'block_1_1')), 'wb') as handle:
            pickle.dump(save_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    wandb.finish()

wandb.agent(sweep_id, function=functools.partial(main, prime_service), count=1)
