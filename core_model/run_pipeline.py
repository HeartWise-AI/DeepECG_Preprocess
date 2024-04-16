#run pipeline
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
import os
import numpy as np
import pprint 
import json
pp = pprint.PrettyPrinter(indent=4)
from torch.utils.data import DataLoader, Dataset
from lion_pytorch import Lion
import traceback
import sys
from collections import defaultdict

from torchmetrics.classification import BinaryAccuracy, BinaryAUROC
from torchmetrics.classification import MultilabelAccuracy, MultilabelAUROC, MulticlassAUROC, MulticlassAccuracy
from torcheval.metrics import MultilabelAUPRC, BinaryAUPRC, MulticlassAUPRC
from statistics import mean
import torch.optim as optim
import pytorch_warmup as warmup
import gc
import wandb
import functools
import hashlib
import time
from ResNet import *
from vit_pytorch import *
from cross_vit_pytorch import *
from EffcientNet import *
from mamba import Mamba, MambaConfig
from validation_of_datasets import *
from numpy.lib.format import read_magic, _read_array_header
from utils import *
import glob
from autoclip.torch import QuantileClip
from termcolor import colored  

import warnings
warnings.filterwarnings("ignore") #this is for some transformation method
#warnings.filterwarnings("ignore", category=RuntimeWarning)  
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:1024" #this helps with memory issues

class CustomError(Exception):
    """Exception raised for errors that do not match any condition in a series of if statements.

    Attributes:
        message -- explanation of the error
    """

    def __init__(self, message="Condition not met for any given cases"):
        self.message = message
        super().__init__(self.message)

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    return config

def generate_dynamic_name_from_config(config):
    """
    Generates a descriptive name string from all configuration parameters in wandb.config.
    Each key-value pair is joined by '.', and pairs are separated by '_'.

    Args:
        config (dict): The configuration dictionary, e.g., from wandb.config.

    Returns:
        str: A string representing the concatenated configuration.
    """
    # Generate the name parts dynamically for all items in the config
    name_parts = [f"{key}.{value}" for key, value in config.items()]
    
    # Join the parts with '_' to form the final name string
    name_str = "_".join(name_parts)
    
    return name_str

def initialize_model(name, config):
    
    if 'resnet' in name:
        model = ResNet1D(
            input_channels=config.channels,
            architecture=config.architecture,
            activation=config.activation,
            dropout=config.dropout,
            use_batchnorm_padding_before_conv1=config.use_batchnorm_padding_before_conv1,
            use_padding_pooling_after_conv1=config.use_padding_pooling_after_conv1,
            stochastic_depth=config.stochastic_depth,
            kernel_sizes=[int(i) for i in json.loads(config.kernel_sizes)],  # Custom kernel sizes for each stage
            strides=[int(i) for i in json.loads(config.strides)],  # Custom strides for each stage
            use_bias=config.use_bias,
            out_neurons=config.out_neurons,
            out_activation=config.out_activation,
            model_width=config.model_width,
            use_ema=config.use_ema)
    
    elif name == 'vit':
        model = ViT(
            seq_len = 2496,
            patch_size = config.patch_size,
            activation = config.activation,
            num_classes = 77,
            dim = config.dim,
            depth = config.depth,
            heads = config.num_heads,
            mlp_dim = config.mlp_dim,
            dropout = config.dropout,
            emb_dropout = config.emb_dropout)


    elif name == 'cross_vit':
        sm_dim, lg_dim = convert_to_two_var(wandb.config.dim)
        sm_patch_size, lg_patch_size = convert_to_two_var(wandb.config.path_sizes)
        sm_enc_depth, lg_enc_depth = convert_to_two_var(wandb.config.enc_depth)
        sm_enc_dim_head,lg_enc_dim_head = convert_to_two_var(wandb.config.enc_dim_head)

        model = CrossViT(
            seq_len = 2496,
            num_classes = 77,
            sm_dim = sm_dim,
            lg_dim = lg_dim,
            sm_patch_size=sm_patch_size,
            lg_patch_size=lg_patch_size,
            sm_enc_depth=sm_enc_depth,
            lg_enc_depth=lg_enc_depth,
            sm_enc_heads = wandb.config.enc_heads,
            lg_enc_heads = wandb.config.enc_heads,
            sm_enc_mlp_dim = wandb.config.mlp_dim,
            lg_enc_mlp_dim = wandb.config.mlp_dim,
            sm_enc_dim_head = sm_enc_dim_head,
            lg_enc_dim_head = lg_enc_dim_head,
            cross_attn_depth = wandb.config.cross_attn_depth,
            cross_attn_heads = wandb.config.cross_attn_heads,
            cross_attn_dim_head = wandb.config.cross_attn_dim_head,
            depth = wandb.config.depth,
            dropout = wandb.config.dropout,
            emb_dropout = wandb.config.emb_dropout,
            activation = wandb.config.activation,
        )

    elif name == 'efficientnet':

        model = EfficientNet1D(
            variant=config.architecture, 
            dropout_rate=config.dropout,
            activation=config.activation, 
            kernel_sizes=[int(i) for i in json.loads(config.kernel_sizes)], 
            strides=[int(i) for i in json.loads(config.strides)],
            stochastic_depth_prob=config.stochastic_depth,   
            se_ratio=[int(i) for i in json.loads(config.se_ratio)], 
            base_depths=[int(i) for i in json.loads(config.base_depths)], 
            base_channels=[int(i) for i in json.loads(config.base_channels)], 
            expansion_factors=[int(i) for i in json.loads(config.expansion_factors)])

    elif name == 'efficientnetv2':
        model = EfficientNet1DV2(
            variant=config.architecture, 
            dropout_rate=config.dropout,
            activation=config.activation, 
            kernel_sizes=[int(i) for i in json.loads(config.kernel_sizes)], 
            strides=[int(i) for i in json.loads(config.strides)],
            stochastic_depth_prob=config.stochastic_depth,   
            se_ratio=[int(i) for i in json.loads(config.se_ratio)], 
            base_depths=[int(i) for i in json.loads(config.base_depths)], 
            base_channels=[int(i) for i in json.loads(config.base_channels)], 
            expansion_factors=[int(i) for i in json.loads(config.expansion_factors)],
            use_se=config.use_se,
            norm_type=config.norm_type)

    else:
        model = None

    return model

def load_numpy_array(filepath):
    try:
        # Attempt to load the file
        data = np.load(filepath).astype(np.float16) #, mmap_mode='r'
        return data
    except FileNotFoundError:
        # Print a custom error message
        pwd = os.getcwd()
        print(f"It appears the directory for the training set is wrong. You can make sure it's in your current directory ({pwd})")
        return None  # or handle it in some other appropriate way

def init_wandb():

    #load the sweep config
    dict_yaml =load_config('sweep_config_2.yml')

    ###################################
    # General optimisation parameters #
    ###################################

    config={'name': dict_yaml['name'], #name of the run
            'method': dict_yaml['method'], #optimisation method
            'metric': {'goal': dict_yaml['metric']['goal'], 'name': dict_yaml['metric']['name']}} #metric to optimise

    parameters = {'data':  {'values': dict_yaml['data_types']},
                'lr': {'max': dict_yaml['lr'][0], 'min': dict_yaml['lr'][1]},
                'optimiser': {'values': dict_yaml['optimiser']},
                'weight_decay':{'max': dict_yaml['weight_decay'][0], 'min': dict_yaml['weight_decay'][1]},
                'batch_size': {'values': dict_yaml['batch_size']},
                'activation': {'values': dict_yaml['activation']}, #activation default ['leaky_relu','gelu','selu','mish','swish']
                'loss': {'values': dict_yaml['loss']},
                'scheduler': {'values': dict_yaml['scheduler']},
                'ema_value': {'max': dict_yaml['ema_value'][0], 'min': dict_yaml['ema_value'][1]},
                'num_max_epochs': {'values': dict_yaml['num_max_epochs']},
                'apply_label_smoothing': {'values': dict_yaml['apply_label_smoothing']},
                'label_smoothing_factor': {'max': dict_yaml['label_smoothing_factor'][0], 'min': dict_yaml['label_smoothing_factor'][1]},
                'rand_augment': {'max': dict_yaml['augment'][0], 'min': dict_yaml['augment'][1]},
                'aug_function': {'values': dict_yaml['function']},
                'out_neurons': {'values': dict_yaml['out_neurons']},
                'out_activation': {'values': dict_yaml['out_activation']},
                'use_adaptive_clipping': {'values': dict_yaml['use_adaptive_clipping']},
                'use_ema':  {'values': dict_yaml['use_ema']},
                'use_warmup': {'values': dict_yaml['use_warmup']},
                'dropout': {'min': dict_yaml['dropout'][0], 'max': dict_yaml['dropout'][1]}} #min max range default: 


    ##################################
    # Load model-specific parameters #
    ##################################

    #saw this as required to allow flexibility for 
    #the desired model

    if dict_yaml['model_type'] == 'resnet': #for resnets - https://arxiv.org/abs/1512.03385

        resnet_parameters = load_config('resnet_config.yml')
    
        parameters.update({
            #resnet-specific parameters
            'channels': {'values': [resnet_parameters['channels']]}, #default 12
            'architecture': {'values': resnet_parameters['architecture']}, #architectures default ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'resnet200']
            'activation': {'values': dict_yaml['activation']}, #activation default ['leaky_relu','gelu','selu','mish','swish']
            'use_batchnorm_padding_before_conv1': {'values': resnet_parameters['use_batchnorm_padding_before_conv1']}, #inspiration from  https://github.com/ZFTurbo/classification_models_1D/blob/main/classification_models_1D/models/resnet.py
            'use_padding_pooling_after_conv1': {'values': resnet_parameters['use_padding_pooling_after_conv1']}, #inspiration from  https://github.com/ZFTurbo/classification_models_1D/blob/main/classification_models_1D/models/resnet.py
            'stochastic_depth': {'max': dict_yaml['stochastic_depth'][0], 'min': dict_yaml['stochastic_depth'][1]}, #stochastic depth default [0.0, 0.5]
            'kernel_sizes': {'values': resnet_parameters['kernel_sizes']}, #kernel_sizes -> patterns already selected for stability
            'strides': {'values': resnet_parameters['strides']}, #strides patterns arlready selected 
            'use_bias': {'values': resnet_parameters['use_bias']}, #use bias always false
            'norm_type': {'values': resnet_parameters['norm_type']}, #use bias always false
            'model_width': {'values': resnet_parameters['model_width']}}) #modle width

    if dict_yaml['model_type'] ==  'vit': #for original vit  https://arxiv.org/abs/2010.11929
        
        vit_parameters = load_config('vit_1d.yml')

        parameters.update({
            'patch_size': {'values':vit_parameters['patch_size']},
            'dim':{'values':vit_parameters['dim']},
            'depth':{'values':vit_parameters['depth']},
            'num_heads':{'values':vit_parameters['num_heads']},
            'mlp_dim':{'values':vit_parameters['mlp_dim']},
            'dropout':{'max': dict_yaml['dropout'][1], 'min': dict_yaml['dropout'][0]}, #stochastic depth default [0.0, 0.5]
            'emb_dropout':{'min': dict_yaml['stochastic_depth'][1], 'max': dict_yaml['stochastic_depth'][0]}, #stochastic depth default [0.0, 0.5]
        })

    if dict_yaml['model_type'] ==  'cross_vit': #for cross vit https://arxiv.org/abs/2103.14899

        cross_vit_parameters = load_config('crossvit_1d.yml')

        parameters.update({
            'path_sizes': {'values':cross_vit_parameters['path_sizes']},
            'dim':{'values':cross_vit_parameters['dim']},
            'depth':{'values':cross_vit_parameters['depth']},
            'enc_depth':{'values':cross_vit_parameters['enc_depth']},
            'enc_heads':{'values':cross_vit_parameters['enc_heads']},
            'mlp_dim':{'values':cross_vit_parameters['mlp_dim']},
            'cross_attn_depth':{'values':cross_vit_parameters['cross_attn_depth']},
            'cross_attn_heads':{'values':cross_vit_parameters['cross_attn_heads']},
            'enc_dim_head':{'values':cross_vit_parameters['enc_dim_head']},
            'cross_attn_dim_head':{'values':cross_vit_parameters['cross_attn_dim_head']},
            'emb_dropout':{'min': dict_yaml['stochastic_depth'][1], 'max': dict_yaml['stochastic_depth'][0]}, #stochastic depth default [0.0, 0.5]
        })

    if dict_yaml['model_type'] ==  'efficientnet':

        efficientnet_parameters = load_config('efficientnet_config.yml')
        
        parameters.update({
            'architecture': {'values': efficientnet_parameters['architecture']}, #architectures default ['b0','b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7']
            'activation': {'values': dict_yaml['activation']}, #activation default ['leaky_relu','gelu','selu','mish','swish']
            'stochastic_depth': {'max': dict_yaml['stochastic_depth'][0], 'min': dict_yaml['stochastic_depth'][1]}, #stochastic depth default [0.0, 0.5]
            'kernel_sizes': {'values': efficientnet_parameters['kernel_sizes']}, #kernel_sizes -> patterns already selected for stability
            'strides': {'values': efficientnet_parameters['strides']}, #strides patterns arlready selected 
            'expansion_factors': {'values': efficientnet_parameters['expansion_factors']}, #strides patterns arlready selected 
            'base_depths': {'values': efficientnet_parameters['base_depths']}, #strides patterns arlready selected 
            'se_ratio': {'values': efficientnet_parameters['default_se_ratio']}, #strides patterns arlready selected 
            'base_channels': {'values': efficientnet_parameters['base_channels']}}) #strides patterns arlready selected 

    if dict_yaml['model_type'] ==  'mamba':

        mamba_parameters = load_config('mamba_config.yml')
            
        parameters.update({
            'd_model': {'values': mamba_parameters['d_model']}, #d_model
            'n_layers': {'values': mamba_parameters['n_layers']}, 
            'd_conv':  {'values': mamba_parameters['d_conv']}, 
            'bias':  {'values': mamba_parameters['bias']}, 
            'conv_bias':  {'values': mamba_parameters['conv_bias']}, 
            'pscan':  {'values': mamba_parameters['pscan']}, 
            'dt_init': {'values': mamba_parameters['dt_init']},
            'expand_factor':  {'values': mamba_parameters['expand_factor']}, 
            'd_state': {'values': mamba_parameters['d_state']}})

    config.update({'parameters':parameters})

    return config

def hash_string(input_string):
    return hashlib.sha256(input_string.encode()).hexdigest()

def convert_to_two_var(config_var):

    config_var_sm, config_var_lm = str(config_var).split('.')
    config_var_sm = int(config_var_sm.split('(')[-1])
    config_var_lm = int(config_var_lm.split(')')[0])

    return config_var_sm, config_var_lm

def load_model(config, name):
    model = initialize_model(name,config)
    return model


def get_data_offset(npy_file_path):
    with open(npy_file_path, 'rb') as f:
        version = format.read_magic(f)
        shape, fortran_order, dtype = format._read_array_header(f, version)
        offset = f.tell()
    return offset, dtype, shape

def main():

    with wandb.init() as run:
    # Overwrite the random run names chosen by wandb
        with open('/volume/benchmark2/sweep_config_2.yml', 'r') as file:
            pipeline_parameters = yaml.load(file, Loader=yaml.FullLoader)        
        
        if len(pipeline_parameters['gpu_to_use']) > 1:
            os.environ["CUDA_VISIBLE_DEVICES"]= ','.join(str(num) for num in pipeline_parameters['gpu_to_use'])
            DEVICE = 'cuda'
        else:
            DEVICE =  pipeline_parameters['gpu_to_use'][0]
        set_seed(pipeline_parameters['seed'][0])
        name_str = generate_dynamic_name_from_config(wandb.config)
        encoded = hash_string(name_str)

        #this was used for Alexis to tey different scaling strategies
        if pipeline_parameters['use_different_scaling']:
            train_path = dict(zip(pipeline_parameters['data_types'],  pipeline_parameters['train_X_path']))
            val_path = dict(zip(pipeline_parameters['data_types'],    pipeline_parameters['val_X_path']))


        #create a run directory if doesn't exist
        os.makedirs(os.path.join(pipeline_parameters['save_dir'],pipeline_parameters['name']), exist_ok=True) 

        #create a run dir for a local save
        os.makedirs(os.path.join(pipeline_parameters['save_dir'],pipeline_parameters['name'],encoded), exist_ok=True)

        if isinstance(pipeline_parameters['train_X_path'], list) and not pipeline_parameters['use_different_scaling']:
            train_path = pipeline_parameters['train_X_path'][0]
            val_path = pipeline_parameters['val_X_path'][0]

        elif isinstance(pipeline_parameters['train_X_path'], str) and not pipeline_parameters['use_different_scaling']:
            train_path = pipeline_parameters['train_X_path']
            val_path = pipeline_parameters['val_X_path']

        ################
        # Loading data #
        ################

        print(colored('Loading data ...', 'green', 'on_black', attrs=["bold"]))
        print('\t loading',  colored('X_train', 'green', attrs=["underline"]), 'from: ', colored(os.path.join(pipeline_parameters['data_path'],val_path), 'green', attrs=["reverse", "blink"]))
        X_train = load_numpy_array(os.path.join(pipeline_parameters['data_path'],val_path))

        print('\t loading',  colored('X_val', 'green', attrs=["underline"]), 'from:   ', colored(os.path.join(pipeline_parameters['data_path'],val_path), 'green', attrs=["reverse", "blink"]))
        X_val = load_numpy_array(os.path.join(pipeline_parameters['data_path'],val_path))

        print('\t loading',  colored('Y_train', 'green', attrs=["underline"]), 'from: ', colored(os.path.join(pipeline_parameters['data_path'],pipeline_parameters['val_Y_path']), 'green', attrs=["reverse", "blink"]))
        Y_train = load_numpy_array(os.path.join(pipeline_parameters['data_path'],pipeline_parameters['val_Y_path']))

        print('\t loading',  colored('Y_val', 'green', attrs=["underline"]), 'from:   ', colored(os.path.join(pipeline_parameters['data_path'],pipeline_parameters['val_Y_path']), 'green', attrs=["reverse", "blink"]))
        Y_val = load_numpy_array(os.path.join(pipeline_parameters['data_path'],pipeline_parameters['val_Y_path']))

        ###################
        # Label smoothing #
        ###################


        if wandb.config.apply_label_smoothing:
            print('\n')
            print(colored('Smoothing ( factor:', 'green', 'on_black', attrs=["bold"]), colored(str(wandb.config.label_smoothing_factor), 'cyan', 'on_black', attrs=["underline"]),  colored(')', 'green', 'on_black', attrs=["bold"]))            
            Y_train = smooth_labels(Y_train,smoothing=wandb.config.label_smoothing_factor)

        ############
        # Fix Axes #
        ############

        if X_train.shape[-2] == 12:
            print('\n')
            print(colored("Flipping the axis to have B,C,N", "green", attrs=["bold"]))
            init_shape = X_train.shape
            X_train = np.swapaxes(X_train, -2, -1)
            X_val = np.swapaxes(X_val, -2, -1)
            post_shape = X_train.shape
            print('\t shape change:', colored(str(init_shape), "green"), colored('->', "green"),  colored(str(init_shape), "green"))

        ####################
        # validate dataset #
        ####################

        if pipeline_parameters['validate_dataset']:
            print('\n')
            print(colored('Running checks can take ~10 mins', "green", attrs=["bold"]))
            main_dataset_check(X_train,X_val,Y_train,Y_val,(pipeline_parameters['expected_X_shape'][0],pipeline_parameters['expected_X_shape'][1]))

        ################
        # clean labels #
        ################

        if pipeline_parameters['clean_labels']:
            print('\n')
            print(colored("removing undesired columns", "green", attrs=["bold"]))
            print('\t Curent number of labels:', colored(str(len(pipeline_parameters['y_label_names'])), "green"))
            print('\t Removing', colored(str(len(pipeline_parameters['labels_to_remove'])), "green"), 'labels')
            pos_to_drop = list()
            new_label_names = list()
            for pos, item in enumerate(pipeline_parameters['y_label_names']):
                if item in pipeline_parameters['labels_to_remove']:
                    pos_to_drop.append(pos)
                else:
                    new_label_names.append(item)

            Y_train = np.delete(Y_train, pos_to_drop, axis=1)
            Y_val = np.delete(Y_val, pos_to_drop, axis=1)

            print('\t Final number of labels',  colored(Y_val.shape[-1], "green"))

        ######################
        # downsample for vit #
        ######################

        if 'vit' in pipeline_parameters['model_type']:
            print('\n')
            print(colored('Reformatting for VIT', "green", attrs=["bold"]))
            #resammple 
            print(X_train.shape)
            X_train = X_train[:,0:2496,:]
            X_val = X_val[:,0:2496,:]

        #log the model variables
        ########################################
        # log the variables to initilize model #
        ########################################

        if pipeline_parameters['model_type'] == 'resnet':
            log_dict = {
                'name': encoded,
                'architecture': wandb.config.architecture,
                'activation': wandb.config.activation,
                'dropout': wandb.config.dropout,
                'use_batchnorm_padding_before_conv1': wandb.config.use_batchnorm_padding_before_conv1,
                'use_padding_pooling_after_conv1': wandb.config.use_padding_pooling_after_conv1,
                'stochastic_depth': wandb.config.stochastic_depth,
                'kernel_sizes': wandb.config.kernel_sizes,
                'strides': wandb.config.strides,
                'use_bias': wandb.config.use_bias,
                'model_width': wandb.config.model_width,
                'data': wandb.config.data,
                'lr': wandb.config.lr,
                'optimiser': wandb.config.optimiser,
                'weight_decay': wandb.config.weight_decay,
                'batch_size': wandb.config.batch_size,
                'loss': wandb.config.loss,
                'scheduler': wandb.config.scheduler,
                'ema_value': wandb.config.ema_value,
                'apply_label_smoothing': wandb.config.apply_label_smoothing,
                'label_smoothing_factor': wandb.config.label_smoothing_factor,
                'rand_augment': wandb.config.rand_augment,
                'aug_function': wandb.config.aug_function,
                'use_adaptive_clipping': wandb.config.use_adaptive_clipping,
                'use_warmup': wandb.config.use_warmup,
                'norm_type':wandb.config.norm_type,
                }
            
        elif pipeline_parameters['model_type'] == 'vit':
            log_dict = {
                'name': encoded,
                'activation': wandb.config.activation,
                'dropout': wandb.config.dropout,
                'emb_dropout': wandb.config.emb_dropout,
                'data': wandb.config.data,
                'lr': wandb.config.lr,
                'optimiser': wandb.config.optimiser,
                'weight_decay': wandb.config.weight_decay,
                'batch_size': wandb.config.batch_size,
                'loss': wandb.config.loss,
                'scheduler': wandb.config.scheduler,
                'ema_value': wandb.config.ema_value,
                'apply_label_smoothing': wandb.config.apply_label_smoothing,
                'label_smoothing_factor': wandb.config.label_smoothing_factor,
                'rand_augment': wandb.config.rand_augment,
                'aug_function': wandb.config.aug_function,
                'use_adaptive_clipping': wandb.config.use_adaptive_clipping,
                'use_warmup': wandb.config.use_warmup,
                'patch_size': wandb.config.patch_size,
                'dim': wandb.config.dim,
                'depth': wandb.config.depth,
                'num_heads': wandb.config.num_heads,
                'mlp_dim': wandb.config.mlp_dim,
                }

        elif pipeline_parameters['model_type'] == 'cross_vit':
            log_dict = {
                'name': encoded,
                'activation': wandb.config.activation,
                'dropout': wandb.config.dropout,
                'emb_dropout': wandb.config.emb_dropout,
                'data': wandb.config.data,
                'lr': wandb.config.lr,
                'optimiser': wandb.config.optimiser,
                'weight_decay': wandb.config.weight_decay,
                'batch_size': wandb.config.batch_size,
                'loss': wandb.config.loss,
                'scheduler': wandb.config.scheduler,
                'ema_value': wandb.config.ema_value,
                'apply_label_smoothing': wandb.config.apply_label_smoothing,
                'label_smoothing_factor': wandb.config.label_smoothing_factor,
                'rand_augment': wandb.config.rand_augment,
                'aug_function': wandb.config.aug_function,
                'use_adaptive_clipping': wandb.config.use_adaptive_clipping,
                'use_warmup': wandb.config.use_warmup,
                'dim': wandb.config.dim,
                'path_sizes': wandb.config.path_sizes,
                'enc_depth': wandb.config.enc_depth,
                'enc_dim_head': wandb.config.enc_dim_head,
                'enc_heads' :  wandb.config.enc_heads,
                'mlp_dim': wandb.config.mlp_dim,
                'cross_attn_depth': wandb.config.cross_attn_depth,
                'cross_attn_heads': wandb.config.cross_attn_heads,
                'cross_attn_dim_head': wandb.config.cross_attn_dim_head,
                'depth': wandb.config.depth,
                }

        elif pipeline_parameters['model_type'] == 'efficientnet' or pipeline_parameters['model_type'] == 'efficientnetv2' :
            log_dict = {
                'name': encoded,
                'expansion_factors': wandb.config.expansion_factors,
                'base_depths': wandb.config.base_depths,
                'use_se': wandb.config.use_se,
                'se_ratio': wandb.config.se_ratio,
                'base_channels': wandb.config.base_channels,
                'architecture': wandb.config.architecture,
                'activation': wandb.config.activation,
                'dropout': wandb.config.dropout,
                'stochastic_depth': wandb.config.stochastic_depth,
                'kernel_sizes': wandb.config.kernel_sizes,
                'strides': wandb.config.strides,
                'data': wandb.config.data,
                'lr': wandb.config.lr,
                'optimiser': wandb.config.optimiser,
                'weight_decay': wandb.config.weight_decay,
                'batch_size': wandb.config.batch_size,
                'loss': wandb.config.loss,
                'scheduler': wandb.config.scheduler,
                'ema_value': wandb.config.ema_value,
                'use_ema': wandb.config.use_ema,
                'apply_label_smoothing': wandb.config.apply_label_smoothing,
                'label_smoothing_factor': wandb.config.label_smoothing_factor,
                'rand_augment': wandb.config.rand_augment,
                'aug_function': wandb.config.aug_function,
                'use_adaptive_clipping': wandb.config.use_adaptive_clipping,
                'use_warmup': wandb.config.use_warmup,
                'ema_value': wandb.config.ema_value,
                'norm_type': wandb.config.norm_type,

                }

        elif pipeline_parameters['model_type'] == 'mamba':
            log_dict = {
                'name': encoded,
                'dropout': wandb.config.dropout,
                'data': wandb.config.data,
                'lr': wandb.config.lr,
                'optimiser': wandb.config.optimiser,
                'weight_decay': wandb.config.weight_decay,
                'batch_size': wandb.config.batch_size,
                'loss': wandb.config.loss,
                'scheduler': wandb.config.scheduler,
                'ema_value': wandb.config.ema_value,
                'rand_augment': wandb.config.rand_augment,
                'aug_function': wandb.config.aug_function,
                'use_adaptive_clipping': wandb.config.use_adaptive_clipping,
                'use_warmup': wandb.config.use_warmup,
                'd_model':wandb.config.d_model,
                'n_layers':wandb.config.n_layers,
                'd_state':wandb.config.d_state,
                'expand_factor':wandb.config.expand_factor,
                'd_conv':wandb.config.d_conv,
                'dt_init':wandb.config.dt_init,
                'bias':wandb.config.bias,
                'conv_bias':wandb.config.conv_bias,
                'pscan':wandb.config.pscan,

                }

        else:
            str_name = pipeline_parameters['model_type']
            raise CustomError(f"Wrong model_type: {str_name}")
            pass
        
        wandb.log(log_dict)

        ##############################################
        # create model or load weights of pretrained #
        ##############################################  
        try:
            if not pipeline_parameters['use_pretrained']:
                model = load_model(wandb.config, pipeline_parameters['model_type'])

            else:
                model = torch.load(pipeline_parameters['pretrained_weights']) #if on 229
        
        ###############################################
        # If fails just clean memory and return error #
        ###############################################
        except:
            raise CustomError(f"Something went wrong when creating/loading the model")
            del X_train
            del Y_train
            del X_val
            del Y_val
            gc.collect()
            torch.cuda.empty_cache()

        #############
        # Optimiser #
        #############
        print('\n')
        if wandb.config.optimiser == 'SGD':
            opt = optim.SGD(model.parameters(), lr=wandb.config.lr, weight_decay=wandb.config.weight_decay)

        elif wandb.config.optimiser == 'RMSPROP':
            opt = optim.RMSprop(model.parameters(), lr=wandb.config.lr, weight_decay=wandb.config.weight_decay)

        elif wandb.config.optimiser == 'Lion':
            opt = Lion(model.parameters(), lr=wandb.config.lr, weight_decay=wandb.config.weight_decay)

        elif wandb.config.optimiser == 'Adam':
            opt = optim.Adam(model.parameters(), lr=wandb.config.lr, weight_decay=wandb.config.weight_decay)

        elif wandb.config.optimiser == 'Radam':
            opt = optim.RAdam(model.parameters(), lr=wandb.config.lr, weight_decay=wandb.config.weight_decay)

        elif wandb.config.optimiser == 'AdamW':
            opt = optim.AdamW(model.parameters(), lr=wandb.config.lr, weight_decay=wandb.config.weight_decay)

        elif wandb.config.optimiser == 'Adagrad':
            opt = optim.Adagrad(model.parameters(), lr=wandb.config.lr, weight_decay=wandb.config.weight_decay)

        elif wandb.config.optimiser == 'NAdam':
            opt = optim.NAdam(model.parameters(), lr=wandb.config.lr, weight_decay=wandb.config.weight_decay)

        elif wandb.config.optimiser == 'LBFGS':
            opt = optim.LBFGS(model.parameters(), lr=wandb.config.lr, weight_decay=wandb.config.weight_decay)

        elif wandb.config.optimiser == 'Adamax':
            opt = optim.Adamax(model.parameters(), lr=wandb.config.lr, weight_decay=wandb.config.weight_decay)

        elif wandb.config.optimiser == 'ASGD':
            opt = optim.ASGD(model.parameters(), lr=wandb.config.lr, weight_decay=wandb.config.weight_decay)

        else:
            print(wandb.config.optimiser)
            raise CustomError(f'{wandb.config.optimiser} selection is not an optimiser choice')
            # print(f'Current {wandb.config.optimiser} selection is not an optimiser choice')

        if  wandb.config.use_adaptive_clipping: #should be much better see the paper: https://arxiv.org/pdf/2007.14469.pdf
            opt = QuantileClip.as_optimizer(optimizer=opt, quantile=0.9, history_length=1000)

        ########
        # LOSS #
        ########

        if pipeline_parameters['training_context'] == 'multilabel':

            if wandb.config.loss == 'binary_ce':
                criterion = nn.BCEWithLogitsLoss()

            elif wandb.config.loss == 'weigthed_binary_crossentropy':
                _, class_counts = np.unique(data.y.items, return_counts=True)
                weights=class_counts/np.sum(class_counts)
                criterion = nn.BCEWithLogitsLoss(pos_weight=weights)

            elif wandb.config.loss == 'MultiLabelSoftMarginLoss':
                criterion = nn.MultiLabelSoftMarginLoss()

            elif wandb.config.loss == 'binary_focalloss_2': #require sigmoid
                kwargs = {"alpha": 0.25, "gamma": 2.0, "reduction": 'mean'}
                criterion = BinaryFocalLossWithLogits(**kwargs)

            elif wandb.config.loss == 'binary_focalloss_3':  #require sigmoid
                kwargs = {"alpha": 0.25, "gamma": 3.0, "reduction": 'mean'}
                criterion = BinaryFocalLossWithLogits(**kwargs)

            elif wandb.config.loss == 'TwoWayLoss':
                criterion = TwoWayLoss()

            elif wandb.config.loss == 'asymmetric_loss':
                criterion = AsymmetricLossOptimized()

            elif wandb.config.loss == 'Hill':
                criterion = Hill()

            elif wandb.config.loss == 'SPLC':
                criterion = SPLC()

            else:
                task_str = pipeline_parameters['training_context']
                print(f'{wandb.config.loss} is not a possible choice of loss for {task_str} tasks')

        elif pipeline_parameters['training_context'] == 'multiclass':
            if wandb.config.loss == 'CCE':
                criterion = nn.CrossEntropyLoss()

            elif wandb.config.loss == 'focalloss':
                criterion = FocalLossMulticlass()
            
            elif wandb.config.loss == 'dice':
                criterion = DiceLoss()

            elif wandb.config.loss == 'tverskyloss':
                criterion = TverskyLoss()

            elif wandb.config.loss == 'FocalCosineLoss':
                criterion = FocalCosineLoss()

            else:
                print(f'Current {wandb.config.optimiser} selection is not an optimiser choice')

        elif pipeline_parameters['training_context'] == 'multiclass':
            if wandb.config.loss == 'binary_ce':
                criterion = nn.BCEWithLogitsLoss()

            elif wandb.config.loss == 'weigthed_binary_crossentropy':
                _, class_counts = np.unique(data.y.items, return_counts=True)
                weights=class_counts/np.sum(class_counts)
                criterion = nn.BCEWithLogitsLoss(pos_weight=weights)

            elif wandb.config.loss == 'MultiLabelSoftMarginLoss':
                criterion = nn.MultiLabelSoftMarginLoss()

            elif wandb.config.loss == 'binary_focalloss_2': #require sigmoid
                kwargs = {"alpha": 0.25, "gamma": 2.0, "reduction": 'mean'}
                criterion = BinaryFocalLossWithLogits(**kwargs)

            elif wandb.config.loss == 'binary_focalloss_3':  #require sigmoid
                kwargs = {"alpha": 0.25, "gamma": 3.0, "reduction": 'mean'}
                criterion = BinaryFocalLossWithLogits(**kwargs)

            else:
                task_str = pipeline_parameters['training_context']
                print(f'{wandb.config.loss} is not a possible choice of loss for {task_str} tasks')
        else:
            task_str = pipeline_parameters['training_context']
            raise CustomError(f'Current {task_str} selection is not a task')
            
        train_params = {'batch_size': wandb.config.batch_size,
                'shuffle': True,
                'num_workers': 12,
                'pin_memory':True}

        val_params = {'batch_size': wandb.config.batch_size,
                'shuffle': False,
                'num_workers': 12,
                'pin_memory':True}

        ##############
        # Generators #
        ##############

        if wandb.config.rand_augment != False:
            training_set = ECGDataset(X_train, Y_train)            
            training_generator = torch.utils.data.DataLoader(training_set, **train_params) #, collate_fn=lambda batch: custom_collate_fn(batch, wandb.config.aug_function , wandb.config.rand_augment))

        else:
            training_set = ECGDataset(X_train, Y_train)
            training_generator = torch.utils.data.DataLoader(training_set, **train_params, collate_fn=lambda batch: val_collate_fn(batch))

        eval_set = ECGDataset(X_val, Y_val)
        eval_generator = torch.utils.data.DataLoader(eval_set, **val_params)
        
        ###########
        # METRICS #
        ###########
        metrics_dict = {
                        'multilabel': {
                            'accuracy': {
                                'micro': MultilabelAccuracy(num_labels=pipeline_parameters['out_neurons'][0], average='micro', multidim_average='global').to(DEVICE),
                                'macro': MultilabelAccuracy(num_labels=pipeline_parameters['out_neurons'][0], average='macro', multidim_average='global').to(DEVICE),
                                'per_class': MultilabelAccuracy(num_labels=pipeline_parameters['out_neurons'][0], average=None).to(DEVICE)
                            },
                            'auroc': {
                                'micro': MultilabelAUROC(num_labels=pipeline_parameters['out_neurons'][0], average='micro').to(DEVICE),
                                'macro': MultilabelAUROC(num_labels=pipeline_parameters['out_neurons'][0], average='macro').to(DEVICE),
                                'per_class': MultilabelAUROC(num_labels=pipeline_parameters['out_neurons'][0], average=None).to(DEVICE)
                            },
                            'auprc': {
                                'macro': MultilabelAUPRC(num_labels=pipeline_parameters['out_neurons'][0], average='macro', device=DEVICE),
                                'per_class': MultilabelAUPRC(num_labels=pipeline_parameters['out_neurons'][0], average=None, device=DEVICE)
                            }
                        },
                        'multiclass': {
                            'accuracy': MulticlassAccuracy(num_classes=pipeline_parameters['out_neurons'][0]).to(DEVICE),
                            'auroc': MulticlassAUROC(num_classes=pipeline_parameters['out_neurons'][0]).to(DEVICE),
                            'auprc': MulticlassAUPRC(num_classes=pipeline_parameters['out_neurons'][0], device=DEVICE)
                        },
                        'binary': {
                            'accuracy': BinaryAccuracy().to(DEVICE),
                            'auroc': BinaryAUROC().to(DEVICE),
                            'auprc': BinaryAUPRC(device=DEVICE)
                        }
                    }

        #LR scheduler
        #scheduler: ['by_plateau'] #by_plateau, cosine_annealing, cyclic, lambda

        if wandb.config.scheduler == 'by_plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.1, patience=1, verbose=True)

        if wandb.config.scheduler == 'cosine_annealing':
            #num_steps = 2000 * 3
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=4000, T_mult=5)

        if wandb.config.scheduler == 'lambda':
            lambda1 = lambda epoch: 0.65 ** wandb.config.num_max_epochs
            scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda1)

        if wandb.config.scheduler == 'triangular2':   
            scheduler = torch.optim.lr_scheduler.CyclicLR(opt, base_lr=wandb.config.lr, max_lr=0.01,step_size_up=4000,mode="triangular2", cycle_momentum=False)

        if wandb.config.scheduler == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(opt, gamma=0.7, step_size=3000)

        if wandb.config.use_warmup == True:
            warmup_scheduler = warmup.ExponentialWarmup(opt,warmup_period=500)

        #use for float16 encoding
        if pipeline_parameters['use_AMP']:
            scaler = torch.cuda.amp.GradScaler()

        print_steps = int(pipeline_parameters['print_every_n_steps'])
        if pipeline_parameters['clean_labels'] == False:
            label_names = pipeline_parameters['y_label_names']
        else:
            label_names = new_label_names

        if len(pipeline_parameters['gpu_to_use']) > 1:
            model = nn.DataParallel(model)

        model = model.to(DEVICE)
        beat_val_loss = 100000000
        best_metric = -1
        best_loss = np.inf
        list_val_loss = list()
        patience = 3
        patience_counter = 0

        if wandb.config.use_ema:
            ema_updater = EMAWeightUpdater(model, decay=wandb.config.ema_value)
        
        print('\n')
        for epoch in range(wandb.config.num_max_epochs):
            model.train()
            train_metrics = defaultdict(list)
            #step accumulators 
            for iter_, (data, label) in enumerate(tqdm(training_generator)):
                data = data.to(DEVICE)
                label = label.to(DEVICE)

                if pipeline_parameters['use_AMP']:
                    with torch.cuda.amp.autocast():
                        if wandb.config.loss == 'SPLC':
                            output = model(data)
                            output = torch.nan_to_num(output)
                            loss = criterion(output, label, epoch)
                        else:
                            output = model(data)
                            output = torch.nan_to_num(output)    
                            loss = criterion(output, label.float())

                        scaler.scale(loss).backward()
                        scaler.step(opt)
                        scaler.update()
                else:
                    if wandb.config.loss == 'SPLC':
                        output = model(data)
                        output = torch.nan_to_num(output)
                        loss = criterion(output, label, epoch)
                    else:
                        output = model(data)
                        output = torch.nan_to_num(output)    
                        loss = criterion(output, label.float())

                    loss.backward()
                    opt.step()


                label = (label > 0.5).long()

                # Update train metrics
                if pipeline_parameters['training_context'] == 'multilabel':
                    for metric_name in ['accuracy','auroc','auprc']:
                        if metric_name in ['accuracy','auroc']:
                            for avg_type in ['micro', 'macro', 'per_class']:
                                metric_value = metrics_dict[pipeline_parameters['training_context']][metric_name][avg_type](output, label)
                                if avg_type == 'per_class':
                                    metric_value = metric_value.detach().cpu().tolist()
                                else:
                                    metric_value = metric_value.detach().cpu().item()

                                train_metrics[f'{metric_name}_{avg_type}'].append(metric_value)
                        
                        if metric_name == 'auprc':
                            for avg_type in ['macro', 'per_class']:
                                metrics_dict[pipeline_parameters['training_context']][metric_name][avg_type].update(output, label)
                                metric_value = metrics_dict[pipeline_parameters['training_context']][metric_name][avg_type].compute()
                                if avg_type == 'per_class':
                                    metric_value = metric_value.detach().cpu().tolist()
                                else:
                                    metric_value = metric_value.detach().cpu().item()
                                train_metrics[f'{metric_name}_{avg_type}'].append(metric_value)  

                if pipeline_parameters['training_context'] in ['multiclass','binary']:
                    for metric_name in ['accuracy','auroc', 'auprc']:
                        if metric_name != 'auprc':
                            metric_value = metrics_dict[pipeline_parameters['training_context']][metric_name](output, label).detach().cpu().item()
                        else:
                            metrics_dict[pipeline_parameters['training_context']][metric_name].update(output, label)
                            metric_value = metrics_dict[pipeline_parameters['training_context']][metric_name].compute().detach().cpu().item()

                        train_metrics[f'{metric_name}'].append(metric_value)


                current_lr = opt.param_groups[0]['lr']

                if iter_ % print_steps == 0 and iter_ != 0:

                    log_dict = {
                        'model': pipeline_parameters['model_type'],
                        'lr': current_lr,
                        'epoch': epoch + 1,
                        'step': iter_
                    }
                    print('\n')
                    print(f'Epoch {epoch+1} iter {iter_}')
                    for metric_name, metric_values in train_metrics.items():
                        if 'per_class' not in metric_name:
                            metric_avg = np.mean(metric_values[iter_ - print_steps:iter_])
                            log_dict[f'train_{metric_name}'] = metric_avg
                            print(colored(metric_name, 'green', attrs=['reverse']), colored(f": {metric_avg:.4f}", 'cyan', attrs=['bold']))
                    print('\n')

                    if bool(pipeline_parameters['print_per_class'] and pipeline_parameters['training_context'] == 'multilabel'):
                        print('**Per class train metrics**')
                        for i, label_name in enumerate(pipeline_parameters['y_label_names']):
                            label_name = label_name.ljust(60)
                            for metric_name in ['accuracy', 'auroc', 'auprc']:
                                metric_values = train_metrics[f'{metric_name}_per_class']
                                metric_avg = np.mean(metric_values[iter_ - print_steps:iter_], axis=0)[i]
                                print(f'\t{label_name} {metric_name}: {metric_avg:.3f}')

                    if bool(pipeline_parameters['log_wandb_performance']):
                        wandb.log(log_dict)

                if wandb.config.use_warmup:
                    with warmup_scheduler.dampening():
                        if wandb.config.scheduler != 'by_plateau' and wandb.config.scheduler != 'none':
                            scheduler.step()
                else:
                    if wandb.config.scheduler != 'by_plateau' and wandb.config.scheduler != 'none':
                        scheduler.step()

            
            #this is for cleanup an preventing memory issues
            del loss
            del output
            del data
            del label

            model.eval()
            if wandb.config.use_ema:
                original_state_dict = ema_updater.apply_shadow()

            with torch.no_grad():

                val_metrics = defaultdict(list)
                
                for val_data, val_label in tqdm(eval_generator):
                    val_data = val_data.to(DEVICE)
                    if wandb.config.loss not in ['binary_ce', 'weigthed_binary_crossentropy']:
                        val_label = val_label.type(torch.int32).to(DEVICE)
                    else:
                        val_label = val_label.type(torch.long).to(DEVICE)

                    if pipeline_parameters['use_AMP']:
                        with torch.cuda.amp.autocast():
                            if wandb.config.loss == 'SPLC':
                                val_output = model(val_data)
                                val_output = torch.nan_to_num(val_output)                   
                                val_loss = criterion(val_output, val_label, epoch)
                            else:
                                val_output = model(val_data)
                                val_output = torch.nan_to_num(val_output)                   
                                val_loss = criterion(val_output, val_label.float())
                    else:
                        if wandb.config.loss == 'SPLC':
                            val_output = model(val_data)
                            val_output = torch.nan_to_num(val_output)                   
                            val_loss = criterion(val_output, val_label, epoch)
                        else:
                            val_output = model(val_data)
                            val_output = torch.nan_to_num(val_output)                   
                            val_loss = criterion(val_output, val_label.float())

                    # Update validation metrics
                    val_metrics['loss'].append(val_loss.detach().cpu().item())
                    # Update val metrics
                    if pipeline_parameters['training_context'] == 'multilabel':
                        for metric_name in ['accuracy','auroc','auprc']:
                            if metric_name in ['accuracy','auroc']:
                                for avg_type in ['micro', 'macro', 'per_class']:
                                    metric_value = metrics_dict[pipeline_parameters['training_context']][metric_name][avg_type](val_output, val_label)
                                    if avg_type == 'per_class':
                                        metric_value = metric_value.detach().cpu().tolist()
                                    else:
                                        metric_value = metric_value.detach().cpu().item()

                                    val_metrics[f'{metric_name}_{avg_type}'].append(metric_value)
                            
                            if metric_name == 'auprc':
                                for avg_type in ['macro', 'per_class']:
                                    metrics_dict[pipeline_parameters['training_context']][metric_name][avg_type].update(val_output, val_label)
                                    metric_value = metrics_dict[pipeline_parameters['training_context']][metric_name][avg_type].compute().detach().cpu()
                                    if avg_type == 'per_class':
                                        metric_value = metric_value.detach().tolist()
                                    else:
                                        metric_value = metric_value.detach().item()
                                    val_metrics[f'{metric_name}_{avg_type}'].append(metric_value)  

                    if pipeline_parameters['training_context'] in ['multiclass','binary']:
                        for metric_name in ['accuracy','auroc', 'auprc']:
                            if metric_name != 'auprc':
                                metric_value = metrics_dict[pipeline_parameters['training_context']][metric_name](val_output, val_label).detach().cpu().item()
                            else:
                                metrics_dict[pipeline_parameters['training_context']][metric_name].update(val_output, val_label)
                                metric_value = metrics_dict[pipeline_parameters['training_context']][metric_name].compute().detach().cpu().item()

                            val_metrics[f'{metric_name}'].append(metric_value)

                # Print validation results
                print('\n')
                print(f"Val E:{epoch+1} Loss ({wandb.config.loss}): {np.mean(val_metrics['loss']):.3f}")
                for metric_name, metric_values in val_metrics.items():
                    if 'per_class' not in metric_name:
                        metric_avg = np.mean(metric_values)
                        log_dict[f'train_{metric_name}'] = metric_avg
                        print(colored(metric_name, 'cyan', attrs=['reverse']), colored(f": {metric_avg:.4f}", 'green', attrs=['bold']))
                print('\n')

                if bool(pipeline_parameters['print_per_class']):
                    print('**Per class val metrics**')
                    for i, label_name in enumerate(pipeline_parameters['y_label_names']):
                        label_name = label_name.ljust(60)
                        for metric_name in ['accuracy', 'auroc', 'auprc']:
                            metric_avg = np.mean(val_metrics[f'{metric_name}_per_class'], axis=0)[i]
                            print(f'{label_name} {metric_name}: {metric_avg:.3f}')

                if np.mean(val_metrics['auprc_macro']) > best_metric:
                    best_metric = np.mean(val_metrics['auprc_macro'])

                if bool(pipeline_parameters['log_wandb_performance']):
                    log_dict = {
                        'epoch': epoch + 1,
                        'best_val_auprc_macro': best_metric
                    }
                    for metric_name, metric_values in val_metrics.items():
                        metric_avg = np.mean(metric_values)
                        log_dict[f'val_{metric_name}'] = metric_avg
                    wandb.log(log_dict)

                if wandb.config.scheduler == 'by_plateau':
                    scheduler.step(np.mean(val_metrics['loss']))

                del val_output
                del val_label
                del val_data

            if wandb.config.use_ema:
                ema_updater.restore_original(original_state_dict)
            val_loss_avg = np.mean(val_metrics['loss'])
            list_val_loss.append(val_loss_avg)

            if val_loss_avg < best_loss:
                best_loss = val_loss_avg
                patience_counter = 0  # reset the patience counter
            else:
                patience_counter += 1  # increase the patience counter

            # Check for early stopping condition
            if patience_counter >= patience:
                print(f"Stopping early at epoch {epoch}.")
                break

            # Save model if improvement
            if val_loss_avg < beat_val_loss:
                print(f'Saving model {val_loss_avg} better than {beat_val_loss}')
                if pipeline_parameters['clean_dir']:
                    for f in glob.glob(os.path.join(pipeline_parameters['save_dir'], pipeline_parameters['name'], encoded, '*')):
                        try:
                            if os.path.isfile(f):  # Ensure it's a file, not a directory
                                os.remove(f)
                        except:
                            pass

                torch.save(model, os.path.join(pipeline_parameters['save_dir'], pipeline_parameters['name'], encoded, f'{pipeline_parameters["model_type"]}_{val_loss_avg}_{wandb.config.loss}_{epoch}.h5'))
                if wandb.config.use_ema:
                    original_state_dict = ema_updater.apply_shadow()  # Apply EMA weights to save them as well
                torch.save(model, os.path.join(pipeline_parameters['save_dir'], pipeline_parameters['name'], encoded, f'{pipeline_parameters["model_type"]}_{val_loss_avg}_{wandb.config.loss}_{epoch}_EMA.h5'))

                if wandb.config.use_ema:
                    ema_updater.restore_original(original_state_dict)

                beat_val_loss = val_loss_avg
            else:
                print(f'Not saving model {val_loss_avg} not better than {beat_val_loss}')


    time.sleep(5)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return 0

# select a model to run
sweep_configuration = init_wandb()
print(sweep_configuration)
pp.pprint(sweep_configuration)
sweep_id = wandb.sweep(sweep=sweep_configuration, project='recent_project') #, entity='mhi_ai') #entity='mhi_ai',


#import pickle

"""

with open('sweep_id_mamba_w_AMP.pickle', 'wb') as handle:
    pickle.dump(sweep_id, handle, protocol=pickle.HIGHEST_PROTOCOL)
"""
#with open('sweep_id_mamba_w_AMP.pickle', 'rb') as handle:
#    sweep_id = pickle.load(handle)

wandb.agent(sweep_id, function=functools.partial(main), count=100)
