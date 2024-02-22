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
from torchmetrics.classification import MultilabelAccuracy, MultilabelAUROC
from torcheval.metrics import MultilabelAUPRC
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

from validation_of_datasets import *

from utils import *
import glob
from autoclip.torch import QuantileClip
from termcolor import colored  

import warnings
warnings.filterwarnings("ignore") #this is for some transformation method
#warnings.filterwarnings("ignore", category=RuntimeWarning) 
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:764"

def init_wandb():

    with open('/volume/benchmark2/sweep_config.yml', 'r') as file:
        dict_yaml = yaml.load(file, Loader=yaml.FullLoader)


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
                'use_warmup': {'values': dict_yaml['use_warmup']},
                'dropout': {'min': dict_yaml['dropout'][0], 'max': dict_yaml['dropout'][1]}} #min max range default: 

            
    if dict_yaml['model_type'] == 'resnet': 

        with open('/volume/benchmark2/resnet_config.yml', 'r') as file:
            resnet_parameters = yaml.load(file, Loader=yaml.FullLoader)
            
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
                'model_width': {'values': resnet_parameters['model_width']}}) #modle width


    if dict_yaml['model_type'] ==  'vit_1d':
        
        with open('/volume/benchmark2/vit_1d.yml', 'r') as file:
            vit_parameters = yaml.load(file, Loader=yaml.FullLoader)
            
            print(vit_parameters)
            
            parameters.update({
                'patch_size': {'values':vit_parameters['patch_size']},
                'dim':{'values':vit_parameters['dim']},
                'depth':{'values':vit_parameters['depth']},
                'num_heads':{'values':vit_parameters['num_heads']},
                'mlp_dim':{'values':vit_parameters['mlp_dim']},
                'dropout':{'max': dict_yaml['dropout'][1], 'min': dict_yaml['dropout'][0]}, #stochastic depth default [0.0, 0.5]
                'emb_dropout':{'min': dict_yaml['stochastic_depth'][1], 'max': dict_yaml['stochastic_depth'][0]}, #stochastic depth default [0.0, 0.5]
            })

    if dict_yaml['model_type'] ==  'cross_vit':
        with open('/volume/benchmark2/crossvit_1d.yml', 'r') as file:
            cross_vit_parameters = yaml.load(file, Loader=yaml.FullLoader)
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
        with open('/volume/benchmark2/efficientnet_config.yml', 'r') as file:
            efficientnet_parameters = yaml.load(file, Loader=yaml.FullLoader)
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
    
    elif name == 'vit_1d':
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

    else:
        model = None

    return model


def main():

    run = wandb.init()

    with wandb.init() as run:
    # Overwrite the random run names chosen by wandb
        with open('/volume/benchmark2/sweep_config.yml', 'r') as file:
            pipeline_parameters = yaml.load(file, Loader=yaml.FullLoader)        

        #run.name = name_str
        
        DEVICE =  pipeline_parameters['gpu_to_use'][0]
        set_seed(pipeline_parameters['seed'][0])

        if pipeline_parameters['model_type'] ==  'resnet':
            name_str = f'{wandb.config.architecture}_{wandb.config.activation}_{wandb.config.dropout}_{wandb.config.use_batchnorm_padding_before_conv1}_{wandb.config.use_padding_pooling_after_conv1}_ \
                        {wandb.config.use_padding_pooling_after_conv1}_{wandb.config.stochastic_depth}_{wandb.config.kernel_sizes}_{wandb.config.strides}_{wandb.config.use_bias}_ \
                        {wandb.config.model_width}_{wandb.config.data}_{wandb.config.lr}_{wandb.config.optimiser}_{wandb.config.weight_decay}_{wandb.config.batch_size}_{wandb.config.loss}_{wandb.config.scheduler}_ \
                        {wandb.config.ema_value}_{wandb.config.apply_label_smoothing}_{wandb.config.label_smoothing_factor}_{wandb.config.rand_augment}_{wandb.config.aug_function}_{wandb.config.use_adaptive_clipping}_{wandb.config.use_warmup}'
            

        if pipeline_parameters['model_type'] ==  'vit_1d':
            name_str = f'{wandb.config.activation}_{wandb.config.dropout}_{wandb.config.emb_dropout}_{wandb.config.data}_{wandb.config.lr}_{wandb.config.optimiser}_ \
                        {wandb.config.weight_decay}_{wandb.config.batch_size}_{wandb.config.loss}_{wandb.config.scheduler}_{wandb.config.patch_size}_{wandb.config.dim}_{wandb.config.depth}_{wandb.config.num_heads}_{wandb.config.mlp_dim}_{wandb.config.dropout}_{wandb.config.dropout}_\
                        {wandb.config.ema_value}_{wandb.config.apply_label_smoothing}_{wandb.config.label_smoothing_factor}_{wandb.config.rand_augment}_{wandb.config.aug_function}_{wandb.config.use_adaptive_clipping}_{wandb.config.use_warmup}'
       

        if pipeline_parameters['model_type'] ==  'cross_vit':
            name_str = f'{wandb.config.activation}_{wandb.config.dropout}_{wandb.config.emb_dropout}_{wandb.config.data}_{wandb.config.lr}_{wandb.config.optimiser}_{wandb.config.weight_decay}_{wandb.config.batch_size}_{wandb.config.loss}_{wandb.config.scheduler}_ \
                        {wandb.config.ema_value}_{wandb.config.apply_label_smoothing}_{wandb.config.label_smoothing_factor}_{wandb.config.rand_augment}_{wandb.config.aug_function}_{wandb.config.use_adaptive_clipping}_{wandb.config.use_warmup}_ \
                        {wandb.config.dim}_{wandb.config.path_sizes}_{wandb.config.enc_depth}_{wandb.config.enc_dim_head}_{wandb.config.enc_heads}_{wandb.config.mlp_dim}_{wandb.config.cross_attn_depth}_ \
                        {wandb.config.cross_attn_heads}_{wandb.config.cross_attn_dim_head}_{wandb.config.depth}_{wandb.config.dropout}'


        if pipeline_parameters['model_type'] ==  'efficientnet':
            name_str =  f'{wandb.config.architecture}_{wandb.config.activation}_{wandb.config.dropout}_{wandb.config.stochastic_depth}_{wandb.config.kernel_sizes}_{wandb.config.strides}_{wandb.config.base_depths}_{wandb.config.se_ratio}_{wandb.config.base_channels}_ \
                        {wandb.config.expansion_factors}_{wandb.config.data}_{wandb.config.lr}_{wandb.config.optimiser}_ \
                        {wandb.config.weight_decay}_{wandb.config.batch_size}_{wandb.config.loss}_{wandb.config.scheduler}_ \
                        {wandb.config.ema_value}_{wandb.config.apply_label_smoothing}_{wandb.config.label_smoothing_factor}_{wandb.config.rand_augment}_{wandb.config.aug_function}_{wandb.config.use_adaptive_clipping}_{wandb.config.use_warmup}'
            
        encoded = hash_string(name_str)
        train_path = dict(zip(pipeline_parameters['data_types'],pipeline_parameters['train_X_path']))
        val_path = dict(zip(pipeline_parameters['data_types'],pipeline_parameters['val_X_path']))

        #create a run directory if doesn't exist
        os.makedirs(os.path.join(pipeline_parameters['save_dir'],pipeline_parameters['name']), exist_ok=True) 

        #create a run dir for a local save
        os.makedirs(os.path.join(pipeline_parameters['save_dir'],pipeline_parameters['name'],encoded), exist_ok=True)

        #load data
        print('Loading data ...')
        print('\t loading X_train')
        X_train = np.load(os.path.join(pipeline_parameters['main_path'],train_path[wandb.config.data]), mmap_mode='r').astype(np.float16)
        print('\t loading X_val')
        X_val = np.load(os.path.join(pipeline_parameters['main_path'],val_path[wandb.config.data]), mmap_mode='r').astype(np.float16)
        #print('\t loading X_test')
        #X_test = np.load(pipeline_parameters['test_X_path']).astype(np.float16)

        print('\t loading Y_train')
        Y_train = np.load(pipeline_parameters['train_Y_path'], mmap_mode='r').astype(np.float16)
        print('\t loading Y_val')
        Y_val = np.load(pipeline_parameters['val_Y_path'], mmap_mode='r').astype(np.float16)
        #print('\t loading Y_test')
        #Y_test = np.load(pipeline_parameters['test_Y_path']) .astype(np.float16)

        if wandb.config.apply_label_smoothing:
            print('smoothed Y')
            print(Y_train)
            Y_train = label_smooth(Y_train,epsilon=wandb.config.label_smoothing_factor)
            print(Y_train)

        if pipeline_parameters['validate_dataset']:
            print('\n')
            print('Running checks can take ~10 mins')
            #main_dataset_check(X_train,X_val,X_test,Y_train,Y_val,Y_test,(pipeline_parameters['expected_X_shape'][0],pipeline_parameters['expected_X_shape'][1]))
            main_dataset_check(X_train,X_val,Y_train,Y_val,(pipeline_parameters['expected_X_shape'][0],pipeline_parameters['expected_X_shape'][1]))


        if X_train.shape[-1] == 12 and wandb.config.rand_augment == 0.0:
            print(colored("Flipping the axis to have B,C,N", "red", attrs=["bold"]))
            X_train = np.swapaxes(X_train, -2, -1)
            X_val = np.swapaxes(X_val, -2, -1)
            #X_test = np.swapaxes(X_test, -2, -1)

        else:
            pass
            #X_val = np.swapaxes(X_val, -2, -1)


        if pipeline_parameters['clean_labels']:
            pos_to_drop = list()
            new_label_names = list()
            for pos, item in enumerate(pipeline_parameters['y_label_names']):
                if item in pipeline_parameters['labels_to_remove']:
                    pos_to_drop.append(pos)
                else:
                    new_label_names.append(item)

            Y_train = np.delete(Y_train, pos_to_drop, axis=1)
            #print(Y_train.shape)
            Y_val = np.delete(Y_val, pos_to_drop, axis=1)
            #print(Y_val.shape)

        if 'vit' in pipeline_parameters['model_type']:
            #resammple 
            print(X_train.shape)
            X_train = X_train[:,0:2496,:]
            X_val = X_val[:,0:2496,:]
            #X_test = X_test[0:2496,:,:]

        #log the model variables
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
                'use_warmup': wandb.config.use_warmup
                }
            
        elif pipeline_parameters['model_type'] == 'vit_1d':
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

        elif pipeline_parameters['model_type'] == 'efficientnet':
            log_dict = {
                'name': encoded,
                'expansion_factors': wandb.config.expansion_factors,
                'base_depths': wandb.config.base_depths,
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
                'apply_label_smoothing': wandb.config.apply_label_smoothing,
                'label_smoothing_factor': wandb.config.label_smoothing_factor,
                'rand_augment': wandb.config.rand_augment,
                'aug_function': wandb.config.aug_function,
                'use_adaptive_clipping': wandb.config.use_adaptive_clipping,
                'use_warmup': wandb.config.use_warmup,
                }

        else:
            pass
        

        wandb.log(log_dict)
        model = load_model(wandb.config, pipeline_parameters['model_type'])
        #print(X_train.shape)
        #print(wandb.config.patch_size)
        """
        try:
            if not pipeline_parameters['use_pretrained']:
                model = load_model(wandb.config, pipeline_parameters['model_type'])

            else:
                model = torch.load('/media/data1/anolin/best_model/p_715.365665435791_asymmetric_loss_5.h5') #if on 229
        
        except:
            print('exception met')
            del X_train
            del Y_train
            del X_val
            del Y_val
            gc.collect()
            torch.cuda.empty_cache()
        """
        #pick optimiser and initial learning rate
        if wandb.config.optimiser == 'SGD':
            opt = optim.SGD(model.parameters(), lr=wandb.config.lr, weight_decay=wandb.config.weight_decay)

        if wandb.config.optimiser == 'RMSPROP':
            opt = optim.RMSprop(model.parameters(), lr=wandb.config.lr, weight_decay=wandb.config.weight_decay)

        if wandb.config.optimiser == 'Lion':
            opt = Lion(model.parameters(), lr=wandb.config.lr, weight_decay=wandb.config.weight_decay)

        if wandb.config.optimiser == 'Adam':
            opt = optim.Adam(model.parameters(), lr=wandb.config.lr, weight_decay=wandb.config.weight_decay)

        if wandb.config.optimiser == 'Radam':
            opt = optim.RAdam(model.parameters(), lr=wandb.config.lr, weight_decay=wandb.config.weight_decay)

        if wandb.config.optimiser == 'AdamW':
            opt = optim.AdamW(model.parameters(), lr=wandb.config.lr, weight_decay=wandb.config.weight_decay)

        if  wandb.config.use_adaptive_clipping:
            opt = QuantileClip.as_optimizer(optimizer=opt, quantile=0.9, history_length=1000)

        #pick loss
        if wandb.config.loss == 'binary_ce':
            criterion = nn.BCEWithLogitsLoss()

        if wandb.config.loss == 'weigthed_binary_crossentropy':
            _, class_counts = np.unique(data.y.items, return_counts=True)
            weights=class_counts/np.sum(class_counts)
            criterion = nn.BCEWithLogitsLoss(pos_weight=weights)

        if wandb.config.loss == 'MultiLabelSoftMarginLoss':
            criterion = nn.MultiLabelSoftMarginLoss()

        if wandb.config.loss == 'binary_focalloss_2': #require sigmoid
            kwargs = {"alpha": 0.25, "gamma": 2.0, "reduction": 'mean'}
            criterion = BinaryFocalLossWithLogits(**kwargs)

        if wandb.config.loss == 'binary_focalloss_3':  #require sigmoid
            kwargs = {"alpha": 0.25, "gamma": 3.0, "reduction": 'mean'}
            criterion = BinaryFocalLossWithLogits(**kwargs)

        if wandb.config.loss == 'TwoWayLoss':
            criterion = TwoWayLoss()

        if wandb.config.loss == 'asymmetric_loss':
            criterion = AsymmetricLossOptimized()

        if wandb.config.loss == 'Hill':
            criterion = Hill()

        if wandb.config.loss == 'SPLC':
            criterion = SPLC()


        train_params = {'batch_size': wandb.config.batch_size,
                'shuffle': True,
                'num_workers': 18,
                'pin_memory':True}

        val_params = {'batch_size': wandb.config.batch_size,
                'shuffle': False,
                'num_workers': 12,
                'pin_memory':True}

        # Generators train
        if wandb.config.rand_augment != 0.0:
            dataset = MyCustomDataset(X_train, Y_train)

            """
            X_compare, Y_compare = np.load("/media/data1/anolin/X_train_v1.1_subsample_10.npy").astype(np.float16), np.load("/media/data1/anolin/Y_train_v1.1_subsample_10.npy").astype(int)

            pos_to_drop = list()
            new_label_names = list()
            for pos, item in enumerate(pipeline_parameters['y_label_names']):
                if item in pipeline_parameters['labels_to_remove']:
                    pos_to_drop.append(pos)
                else:
                    new_label_names.append(item)
            Y_compare = np.delete(Y_compare, pos_to_drop, axis=1)

            #print(Y_compare.shape)
            #print(Y_train.shape)
            """
            #X_compare = np.divide(X_compare, 5.2)            
            training_generator = torch.utils.data.DataLoader(dataset, **train_params, collate_fn=lambda batch: custom_collate_fn(batch, wandb.config.aug_function , wandb.config.rand_augment))

            #print(pipeline_parameters['function'])3
            #dataset = MyCustomDataset(X_train, Y_train)
            #training_generator = torch.utils.data.DataLoader(dataset, batch_size=wandb.config.batch_size, shuffle=True, num_workers=4, collate_fn=lambda batch: custom_collate_fn(batch, window_warp_multithreaded, 0.1))

        else:
            training_set = ECGDataset(X_train, Y_train)
            training_generator = torch.utils.data.DataLoader(training_set, **train_params)


        # Generators train
        eval_set = ECGDataset(X_val, Y_val)
        eval_generator = torch.utils.data.DataLoader(eval_set, **val_params)

        #metrics

        #grouped metrics
        #accuracy
        MultilabelAccuracy_micro_group = MultilabelAccuracy(num_labels=pipeline_parameters['out_neurons'][0], average='micro', multidim_average='global').to(DEVICE)
        MultilabelAccuracy_macro_group = MultilabelAccuracy(num_labels=pipeline_parameters['out_neurons'][0], average='macro', multidim_average='global').to(DEVICE)

        #AUROC
        MultilabelAUROC_micro_group = MultilabelAUROC(num_labels=pipeline_parameters['out_neurons'][0], average='micro').to(DEVICE)
        MultilabelAUROC_macro_group = MultilabelAUROC(num_labels=pipeline_parameters['out_neurons'][0], average='macro').to(DEVICE)

        #AUPRC
        MultilabelAUPRC_macro = MultilabelAUPRC(num_labels=pipeline_parameters['out_neurons'][0], average='macro',device=DEVICE)

        #individual metrics
        #accuracy
        MultilabelAccuracy_per_class = MultilabelAccuracy(num_labels=pipeline_parameters['out_neurons'][0],average=None).to(DEVICE)

        #AUROC
        MultilabelAUROC_per_class = MultilabelAUROC(num_labels=pipeline_parameters['out_neurons'][0], average=None).to(DEVICE)
        
        #AUPRC
        MultilabelAUPRC_per_class = MultilabelAUPRC(num_labels=pipeline_parameters['out_neurons'][0], average=None,device=DEVICE)

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
        scaler = torch.cuda.amp.GradScaler()

        #epoch average accumulators 

        #train
        #train_epoch_loss = list()
        #train_epoch_MultilabelAccuracy_micro_group = list()
        #train_epoch_MultilabelAccuracy_macro_group = list()

        #train_epoch_MultilabelAUROC_micro_group = list()
        #train_epoch_MultilabelAUROC_macro_group = list()

        #train_epoch_MultilabelAUPRC_macro = list()

        #train_epoch_MultilabelAccuracy_per_class = dict(zip(  list(range(len(pipeline_parameters['y_label_names']))),   [] for i in  range(len(pipeline_parameters['y_label_names']))   ))
        #train_epoch_MultilabelAUROC_per_class = dict(zip(  list(range(len(pipeline_parameters['y_label_names']))),   [] for i in  range(len(pipeline_parameters['y_label_names']))   ))
        #train_epoch_MultilabelAUPRC_per_class = dict(zip(  list(range(len(pipeline_parameters['y_label_names']))),   [] for i in  range(len(pipeline_parameters['y_label_names']))   ))

        
        print_steps = int(pipeline_parameters['print_every_n_steps'])
        if pipeline_parameters['clean_labels'] == False:
            label_names = pipeline_parameters['y_label_names']
        else:
            label_names = new_label_names

        model = model.to(DEVICE)
        beat_val_loss = 100000000
        best_metric = -1
        best_loss = np.inf
        list_val_loss = list()
        patience = 3
        patience_counter = 0
        ema_updater = EMAWeightUpdater(model, decay=wandb.config.ema_value)

        print(X_train.shape)
        print(X_val.shape)

        try:

            for epoch in range(wandb.config.num_max_epochs):
                model.train()
                #step accumulators 

                train_steps_loss = list()
                train_steps_MultilabelAccuracy_micro_group = list()
                train_steps_MultilabelAccuracy_macro_group = list()

                train_steps_MultilabelAUROC_micro_group = list()
                train_steps_MultilabelAUROC_macro_group = list()

                train_steps_MultilabelAUPRC_macro = list()

                train_steps_MultilabelAccuracy_per_class = dict(zip(  list(range(len(pipeline_parameters['y_label_names']))),   [[] for i in  range(len(pipeline_parameters['y_label_names']))]   ))
                train_steps_MultilabelAUROC_per_class = dict(zip(  list(range(len(pipeline_parameters['y_label_names']))),   [[] for i in  range(len(pipeline_parameters['y_label_names']))]   ))
                train_steps_MultilabelAUPRC_per_class = dict(zip(  list(range(len(pipeline_parameters['y_label_names']))),   [[] for i in  range(len(pipeline_parameters['y_label_names']))]   ))


                for iter_, (data, label) in enumerate(tqdm(training_generator)):
                    data = data.to(DEVICE)

                    label = label.type(torch.long).to(DEVICE)

                    if wandb.config.loss == 'SPLC':
                        with torch.autocast(device_type='cuda', dtype=torch.float16):
                            output = model(data)
                            output = torch.nan_to_num(output)
                            loss = criterion(output, label, epoch)

                    else:
                        with torch.autocast(device_type='cuda', dtype=torch.float16):
                            output = model(data)
                            output = torch.nan_to_num(output)                   
                            loss = criterion(output, label.float())

                    opt.zero_grad()
                    scaler.scale(loss).backward()
                
                    scaler.unscale_(opt)
                    #if wandb.config.use_adaptive_clipping:
                    #    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)

                    scaler.step(opt)
                    scaler.update()
                    ema_updater.update()

                    #acquire train metrics
                    train_steps_loss.append(loss.detach().cpu().item())
                    train_steps_MultilabelAccuracy_micro_group.append(MultilabelAccuracy_micro_group(output,label).detach().cpu().item())
                    MultilabelAccuracy_micro_group.reset()
                    train_steps_MultilabelAccuracy_macro_group.append(MultilabelAccuracy_macro_group(output,label).detach().cpu().item())
                    MultilabelAccuracy_macro_group.reset()
                    train_steps_MultilabelAUROC_micro_group.append(MultilabelAUROC_micro_group(output,label).detach().cpu().item())
                    MultilabelAUROC_micro_group.reset()
                    train_steps_MultilabelAUROC_macro_group.append(MultilabelAUROC_macro_group(output,label).detach().cpu().item())
                    MultilabelAUROC_macro_group.reset()
                    
                    MultilabelAUPRC_macro.update(output, label)
                    train_steps_MultilabelAUPRC_macro.append(MultilabelAUPRC_macro.compute().detach().cpu().item())
                    MultilabelAUPRC_macro.reset()

                    #zip the batch results per class
                    MultilabelAccuracy_per_class_results = [item.detach().cpu().item() for item in MultilabelAccuracy_per_class(output,label)]
                    for key, value in zip(train_steps_MultilabelAccuracy_per_class.keys(), MultilabelAccuracy_per_class_results):
                        train_steps_MultilabelAccuracy_per_class[key].append(value)
                    MultilabelAccuracy_per_class.reset()

                    MultilabelAUROC_per_class_results = [item.detach().cpu().item() for item in MultilabelAUROC_per_class(output,label)]
                    for key, value in zip(train_steps_MultilabelAUROC_per_class.keys(), MultilabelAUROC_per_class_results):
                        train_steps_MultilabelAUROC_per_class[key].append(value)
                    MultilabelAUROC_per_class.reset()

                    MultilabelAUPRC_per_class.update(output, label)
                    MultilabelAUPRC_per_class_results = [item.detach().cpu().item() for item in MultilabelAUPRC_per_class.compute()]
                    for key, value in zip(train_steps_MultilabelAUPRC_per_class.keys(), MultilabelAUPRC_per_class_results):
                        train_steps_MultilabelAUPRC_per_class[key].append(value)
                    MultilabelAUPRC_per_class.reset()

                    current_lr = opt.param_groups[0]['lr']

                    if iter_ % print_steps == 0 and iter_ != 0:
                        print(f"Train E:{epoch+1} Step:{iter_} LR {current_lr} Loss {wandb.config.loss}: {'{:.3f}'.format(mean(train_steps_loss[iter_-print_steps:iter_]))} Acc micro {'{:.3f}'.format(mean(train_steps_MultilabelAccuracy_micro_group[iter_-print_steps:iter_]))} Acc macro {'{:.3f}'.format(mean(train_steps_MultilabelAccuracy_macro_group[iter_-print_steps:iter_]))}  ROC micro {'{:.3f}'.format(mean(train_steps_MultilabelAUROC_micro_group[iter_-print_steps:iter_]))} ROC macro {'{:.3f}'.format(mean(train_steps_MultilabelAUROC_macro_group[iter_-print_steps:iter_]))} AUPRC macro {'{:.3f}'.format(mean(train_steps_MultilabelAUPRC_macro[iter_-print_steps:iter_]))}")

                        if bool(pipeline_parameters['print_per_class']):
                            print('**Per class train metrics**')

                            for i in range(len(pipeline_parameters['y_label_names'])):
                                label_name = pipeline_parameters['y_label_names'][i]
                                label_name = label_name.ljust(60)
                                print(f'\t{label_name} Acc {"{:.3f}".format(mean(train_steps_MultilabelAccuracy_per_class[i][iter_-print_steps:iter_]))} ROC {"{:.3f}".format(mean(train_steps_MultilabelAUROC_per_class[i][iter_-print_steps:iter_]))} PR {"{:.3f}".format(mean(train_steps_MultilabelAUPRC_per_class[i][iter_-print_steps:iter_]))}')


                        if bool(pipeline_parameters['log_wandb_performance']):

                            #logg simple stuff
                            log_dict = {
                            'model':'model_to_run',
                            'lr': current_lr,
                            "train_loss": mean(train_steps_loss[iter_-print_steps:iter_]),
                            "train_MultilabelAccuracy_micro":mean(train_steps_MultilabelAccuracy_micro_group[iter_-print_steps:iter_]),
                            "train_MultilabelAccuracy_macro":mean(train_steps_MultilabelAccuracy_macro_group[iter_-print_steps:iter_]),
                            "train_MultilabelAUROC_micro":mean(train_steps_MultilabelAUROC_micro_group[iter_-print_steps:iter_]),
                            "train_MultilabelAUROC_macro":mean(train_steps_MultilabelAUROC_macro_group[iter_-print_steps:iter_]),
                            "train_MultilabelAUPRC_micro":mean(train_steps_MultilabelAUPRC_macro[iter_-print_steps:iter_]),
                            }

                            #iterate for the per class
                            for metric in ['accuracy','AUROC','AUPRC']:                                         
                                for i in range(len(label_names)):
                                    if metric == 'accuracy':
                                        dict_ = train_steps_MultilabelAccuracy_per_class

                                    elif metric == 'AUROC':
                                        dict_ = train_steps_MultilabelAUROC_per_class

                                    else:
                                        dict_ = train_steps_MultilabelAUPRC_per_class

                                    log_dict.update({f'train_{metric}_{label_names[i]}': mean(dict_[i][iter_-print_steps:iter_])})
                            
                            wandb.log(log_dict)

                    if wandb.config.use_warmup:
                        #print('warmup')
                        #print(opt.param_groups[0]['lr'])
                        with warmup_scheduler.dampening():
                            if wandb.config.scheduler != 'by_plateau' and wandb.config.scheduler != 'none':
                                scheduler.step()

                    else:
                        if wandb.config.scheduler != 'by_plateau' and wandb.config.scheduler != 'none':
                            scheduler.step()

        
                del loss
                del output
                del data
                del label

                del train_steps_loss
                del train_steps_MultilabelAccuracy_micro_group
                del train_steps_MultilabelAccuracy_macro_group
                del train_steps_MultilabelAUROC_micro_group
                del train_steps_MultilabelAUROC_macro_group
                del train_steps_MultilabelAUPRC_macro

                model.eval()
                original_state_dict = ema_updater.apply_shadow()

                with torch.no_grad():
                    
                    val_steps_loss = list()
                    val_steps_MultilabelAccuracy_micro_group = list()
                    val_steps_MultilabelAccuracy_macro_group = list()

                    val_steps_MultilabelAUROC_micro_group = list()
                    val_steps_MultilabelAUROC_macro_group = list()

                    val_steps_MultilabelAUPRC_macro = list()

                    val_steps_MultilabelAccuracy_per_class = dict(zip(  list(range(len(pipeline_parameters['y_label_names']))),[[] for i in  range(len(pipeline_parameters['y_label_names']))]   ))
                    val_steps_MultilabelAUROC_per_class = dict(zip(  list(range(len(pipeline_parameters['y_label_names']))),   [[] for i in  range(len(pipeline_parameters['y_label_names']))]   ))
                    val_steps_MultilabelAUPRC_per_class = dict(zip(  list(range(len(pipeline_parameters['y_label_names']))),   [[] for i in  range(len(pipeline_parameters['y_label_names']))]   ))

                    for val_data, val_label in tqdm(eval_generator):
                        val_data = val_data.to(DEVICE)
                        if wandb.config.loss not in ['binary_ce','weigthed_binary_crossentropy']:
                            val_label = val_label.type(torch.int32).to(DEVICE)

                        else:
                            val_label = val_label.type(torch.long).to(DEVICE)

                        if wandb.config.loss == 'SPLC':
                            val_output = model(val_data)
                            val_output = torch.nan_to_num(val_output)                   
                            val_loss = criterion(val_output, val_label, epoch)

                        else:
                            val_output = model(val_data)
                            val_output = torch.nan_to_num(val_output)                   
                            val_loss = criterion(val_output, val_label.float())

                        #acquire train metrics
                        val_steps_loss.append(val_loss.detach().cpu().item())
                        val_steps_MultilabelAccuracy_micro_group.append(MultilabelAccuracy_micro_group(val_output,val_label).detach().cpu().item())
                        MultilabelAccuracy_micro_group.reset()
                        val_steps_MultilabelAccuracy_macro_group.append(MultilabelAccuracy_macro_group(val_output,val_label).detach().cpu().item())
                        MultilabelAccuracy_macro_group.reset()

                        val_steps_MultilabelAUROC_micro_group.append(MultilabelAUROC_micro_group(val_output,val_label).detach().cpu().item())
                        MultilabelAUROC_micro_group.reset()
                        val_steps_MultilabelAUROC_macro_group.append(MultilabelAUROC_macro_group(val_output,val_label).detach().cpu().item())
                        MultilabelAUROC_macro_group.reset()

                        MultilabelAUPRC_macro.update(val_output, val_label)
                        val_steps_MultilabelAUPRC_macro.append(MultilabelAUPRC_macro.compute().detach().cpu().item())
                        MultilabelAUPRC_macro.reset()

                        #zip the batch results per class
                        MultilabelAccuracy_per_class_results = [item.detach().cpu().item() for item in MultilabelAccuracy_per_class(val_output,val_label)]
                        for key, value in zip(val_steps_MultilabelAccuracy_per_class.keys(), MultilabelAccuracy_per_class_results):
                            val_steps_MultilabelAccuracy_per_class[key].append(value)
                        MultilabelAccuracy_per_class.reset()

                        MultilabelAUROC_per_class_results = [item.detach().cpu().item() for item in MultilabelAUROC_per_class(val_output,val_label)]
                        for key, value in zip(val_steps_MultilabelAUROC_per_class.keys(), MultilabelAUROC_per_class_results):
                            val_steps_MultilabelAUROC_per_class[key].append(value)
                        MultilabelAUROC_per_class.reset()

                        MultilabelAUPRC_per_class.update(val_output, val_label)
                        MultilabelAUPRC_per_class_results = [item.detach().cpu().item() for item in MultilabelAUPRC_per_class.compute()]
                        for key, value in zip(val_steps_MultilabelAUPRC_per_class.keys(), MultilabelAUPRC_per_class_results):
                            val_steps_MultilabelAUPRC_per_class[key].append(value)
                        MultilabelAUPRC_per_class.reset()
                    
                    del val_output
                    del val_label
                    del val_data
                    #print the val results
                    print(f"Val E:{epoch+1} Loss ({wandb.config.loss}): {'{:.3f}'.format(mean(val_steps_loss))} Acc micro {'{:.3f}'.format(mean(val_steps_MultilabelAccuracy_micro_group))} Acc macro {'{:.3f}'.format(mean(val_steps_MultilabelAccuracy_macro_group))} ROC micro {'{:.3f}'.format(mean(val_steps_MultilabelAUROC_micro_group))} ROC macro {'{:.3f}'.format(mean(val_steps_MultilabelAUROC_macro_group))} AUPRC macro {'{:.3f}'.format(mean(val_steps_MultilabelAUPRC_macro))}")

                    if bool(pipeline_parameters['print_per_class']):
                        print('**Per class val metrics**')
                        for i in range(len(pipeline_parameters['y_label_names'])):
                            label_name = pipeline_parameters['y_label_names'][i]
                            label_name = label_name.ljust(60)
                            print(f'{label_name} Acc {"{:.3f}".format(mean(val_steps_MultilabelAccuracy_per_class[i]))} ROC {"{:.3f}".format(mean(val_steps_MultilabelAUROC_per_class[i]))} PR {"{:.3f}".format(mean(val_steps_MultilabelAUPRC_per_class[i]))}')


                    if mean(val_steps_MultilabelAUPRC_macro) > best_metric:
                        best_metric = mean(val_steps_MultilabelAUPRC_macro)
            
                    if bool(pipeline_parameters['log_wandb_performance']):

                        #logg simple stuff
                        log_dict = {
                        "val_loss": mean(val_steps_loss),
                        "val_MultilabelAccuracy_micro":mean(val_steps_MultilabelAccuracy_micro_group),
                        "val_MultilabelAccuracy_macro":mean(val_steps_MultilabelAccuracy_macro_group),
                        "val_MultilabelAUROC_micro":mean(val_steps_MultilabelAUROC_micro_group),
                        "val_MultilabelAUROC_macro":mean(val_steps_MultilabelAUROC_macro_group),
                        "val_MultilabelAUPRC_macro":mean(val_steps_MultilabelAUPRC_macro),
                        "best_val_MultilabelAUPRC_macro":best_metric,
                        }

                        #iterate for the per class
                        for metric in ['accuracy','AUROC','AUPRC']:                                         
                            for i in range(len(label_names)):
                                if metric == 'accuracy':
                                    dict_ = val_steps_MultilabelAccuracy_per_class

                                elif metric == 'AUROC':
                                    dict_ = val_steps_MultilabelAUROC_per_class

                                else:
                                    dict_ = val_steps_MultilabelAUPRC_per_class

                                log_dict.update({f'val_{metric}_{label_names[i]}':mean(dict_[i])})

                    if wandb.config.scheduler == 'by_plateau':
                        scheduler.step(mean(val_steps_loss))


                    wandb.log(log_dict)
                
                ema_updater.restore_original(original_state_dict)
                val_steps_loss_ = mean(val_steps_loss)
                list_val_loss.append(val_steps_loss_)

                if val_steps_loss_ < best_loss:
                    best_loss = val_steps_loss_
                    patience_counter = 0  # reset the patience counter
                else:
                    patience_counter += 1  # increase the patience counter

                # Check for early stopping condition
                if patience_counter >= patience:
                    print(f"Stopping early at epoch {epoch}.")
                    break


                #save model if improvement
                if mean(val_steps_loss) < beat_val_loss:
                    print(f'Saving model {mean(val_steps_loss)} bettter than {beat_val_loss}')
                    if pipeline_parameters['clean_dir']:
                        for f in glob.glob(os.path.join(pipeline_parameters['save_dir'],pipeline_parameters['name'],encoded,'*')):
                            try:
                                if os.path.isfile(f):  # Ensure it's a file, not a directory
                                    os.remove(f)
                            except:
                                pass


                    torch.save(model, os.path.join(pipeline_parameters['save_dir'],pipeline_parameters['name'],encoded,f'{pipeline_parameters["model_to_run"]}_{mean(val_steps_loss)}_{wandb.config.loss}_{epoch}.h5'))
                    
                    original_state_dict = ema_updater.apply_shadow()  # Apply EMA weights to save them as well
                    torch.save(model, os.path.join(pipeline_parameters['save_dir'],pipeline_parameters['name'],encoded,f'{pipeline_parameters["model_to_run"]}_{mean(val_steps_loss)}_{wandb.config.loss}_{epoch}_EMA.h5'))
                    ema_updater.restore_original(original_state_dict)
                    beat_val_loss = mean(val_steps_loss)

                else:
                    print(f'Not saving model {mean(val_steps_loss)} not better than {beat_val_loss}')


                if mean(val_steps_MultilabelAUPRC_macro) < 0.15 and  epoch+1 >= 1:
                    print('Performance sucks')
                    time.sleep(60)
                    del val_steps_loss
                    del val_steps_MultilabelAccuracy_micro_group
                    del val_steps_MultilabelAccuracy_macro_group
                    del val_steps_MultilabelAUROC_micro_group
                    del val_steps_MultilabelAUROC_macro_group
                    del val_steps_MultilabelAUPRC_macro
                    del val_loss
                    del X_train
                    del Y_train
                    del X_val
                    del Y_val
                    del model

                    gc.collect()
                    torch.cuda.empty_cache()
                    return 0

                if mean(val_steps_MultilabelAUPRC_macro) < 0.2 and epoch+1 >= 2:
                    print('Performance sucks')
                    time.sleep(60)
                    del val_steps_loss
                    del val_steps_MultilabelAccuracy_micro_group
                    del val_steps_MultilabelAccuracy_macro_group
                    del val_steps_MultilabelAUROC_micro_group
                    del val_steps_MultilabelAUROC_macro_group
                    del val_steps_MultilabelAUPRC_macro
                    del val_loss
                    del X_train
                    del Y_train
                    del X_val
                    del Y_val
                    del model
                   
                    gc.collect()
                    torch.cuda.empty_cache()
                    return 0

                if mean(val_steps_MultilabelAUPRC_macro) < 0.3 and epoch+1 >= 4:
                    print('Performance sucks')
                    time.sleep(60)
                    del val_steps_loss
                    del val_steps_MultilabelAccuracy_micro_group
                    del val_steps_MultilabelAccuracy_macro_group
                    del val_steps_MultilabelAUROC_micro_group
                    del val_steps_MultilabelAUROC_macro_group
                    del val_steps_MultilabelAUPRC_macro
                    del val_loss
                    del X_train
                    del Y_train
                    del X_val
                    del Y_val
                    del model
                    gc.collect()
                    torch.cuda.empty_cache()
                    return 0

                elif mean(val_steps_MultilabelAUPRC_macro) < 0.40 and epoch+1 >= 6:
                    print('Performance sucks')
                    time.sleep(60)
                    del val_steps_loss
                    del val_steps_MultilabelAccuracy_micro_group
                    del val_steps_MultilabelAccuracy_macro_group
                    del val_steps_MultilabelAUROC_micro_group
                    del val_steps_MultilabelAUROC_macro_group
                    del val_steps_MultilabelAUPRC_macro
                    del val_loss
                    del X_train
                    del Y_train
                    del X_val
                    del Y_val
                    del model
                    gc.collect()
                    torch.cuda.empty_cache()
                    return 0

                else:
                    pass

                del val_steps_loss
                del val_steps_MultilabelAccuracy_micro_group
                del val_steps_MultilabelAccuracy_macro_group
                del val_steps_MultilabelAUROC_micro_group
                del val_steps_MultilabelAUROC_macro_group
                del val_steps_MultilabelAUPRC_macro
                del val_loss
                del model
               
                gc.collect()
                torch.cuda.empty_cache()

        
        
        except Exception:
            traceback.print_exc()
            print('bad')
            time.sleep(60)

            del X_train
            del Y_train
            del X_val
            del Y_val
            del model
            gc.collect()
            torch.cuda.empty_cache()
            return 0

    time.sleep(60)
    del X_train
    del Y_train
    del X_val
    del Y_val
    del model
    gc.collect()
    torch.cuda.empty_cache()



# [37, 7, 7, 5, 5]
# [3, 3, 3, 2, 2]

# select a model to run
sweep_configuration = init_wandb()
pp.pprint(sweep_configuration)
sweep_id = wandb.sweep(sweep=sweep_configuration, project='benchmark_efficientnet_2', entity='mhi_ai') #entity='mhi_ai',

import pickle

with open('sweep_id_efficientnet.pickle', 'wb') as handle:
    pickle.dump(sweep_id, handle, protocol=pickle.HIGHEST_PROTOCOL)
"""
with open('sweep_id_efficientnet.pickle', 'rb') as handle:
    sweep_id = pickle.load(handle)
"""

wandb.agent(sweep_id, function=functools.partial(main), count=100)
