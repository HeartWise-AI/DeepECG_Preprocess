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

from torchmetrics.classification import MultilabelAccuracy, MultilabelAUROC
from torcheval.metrics import MultilabelAUPRC
from statistics import mean
import torch.optim as optim
import pytorch_warmup as warmup
import gc
import time
import wandb
import functools
import hashlib
import glob
from ResNet import *
from validation_of_datasets import *

from utils import *

from autoclip.torch import QuantileClip
from termcolor import colored  

import warnings
warnings.filterwarnings("ignore") #this is for some transformation method
#warnings.filterwarnings("ignore", category=RuntimeWarning) 

def init_wandb():

    with open('/volume/benchmark2/sweep_config.yml', 'r') as file:
        dict_yaml = yaml.load(file, Loader=yaml.FullLoader)


    config={'name': dict_yaml['name'], #name of the run
            'method': dict_yaml['method'], #optimisation method
            'metric': {'goal': dict_yaml['metric']['goal'], 'name': dict_yaml['metric']['name']}} #metric to optimise

 
    if dict_yaml['model_to_run'] ==  'pytorch_resnet': 

        with open('/volume/benchmark2/resnet_config.yml', 'r') as file:
            resnet_parameters = yaml.load(file, Loader=yaml.FullLoader)
            
            parameters = {
                #resnet-specific parameters
                'channels': {'values': [resnet_parameters['channels']]}, #default 12
                'architecture': {'values': resnet_parameters['architecture']}, #architectures default ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'resnet200']
                'activation': {'values': dict_yaml['activation']}, #activation default ['leaky_relu','gelu','selu','mish','swish']
                'dropout': {'min': dict_yaml['dropout'][0], 'max': dict_yaml['dropout'][1]}, #min max range default: 
                'use_batchnorm_padding_before_conv1': {'values': resnet_parameters['use_batchnorm_padding_before_conv1']}, #inspiration from  https://github.com/ZFTurbo/classification_models_1D/blob/main/classification_models_1D/models/resnet.py
                'use_padding_pooling_after_conv1': {'values': resnet_parameters['use_padding_pooling_after_conv1']}, #inspiration from  https://github.com/ZFTurbo/classification_models_1D/blob/main/classification_models_1D/models/resnet.py
                'stochastic_depth': {'max': dict_yaml['stochastic_depth'][0], 'min': dict_yaml['stochastic_depth'][1]}, #stochastic depth default [0.0, 0.5]
                'kernel_sizes': {'values': resnet_parameters['kernel_sizes']}, #kernel_sizes -> patterns already selected for stability
                'strides': {'values': resnet_parameters['strides']}, #strides patterns arlready selected 
                'use_bias': {'values': resnet_parameters['use_bias']}, #use bias always false
                'model_width': {'values': resnet_parameters['model_width']}, #modle width


                #general optimization parameters
                'data':  {'values': dict_yaml['data_types']},
                'lr': {'max': dict_yaml['lr'][0], 'min': dict_yaml['lr'][1]},
                'optimiser': {'values': dict_yaml['optimiser']},
                'weight_decay':{'max': dict_yaml['weight_decay'][0], 'min': dict_yaml['weight_decay'][1]},
                'batch_size': {'values': dict_yaml['batch_size']},
                'loss': {'values': dict_yaml['loss']},
                'scheduler': {'values': dict_yaml['scheduler']},
                'num_max_epochs': {'values': dict_yaml['num_max_epochs']},
                'use_ema': {'values': dict_yaml['use_EMA']},
                'apply_label_smoothing': {'values': dict_yaml['apply_label_smoothing']},
                'label_smoothing_factor': {'max': dict_yaml['label_smoothing_factor'][0], 'min': dict_yaml['label_smoothing_factor'][1]},
                'rand_augment': {'max': dict_yaml['augment'][0], 'min': dict_yaml['augment'][1]},
                'aug_function': {'values': dict_yaml['function']},
                'out_neurons': {'values': dict_yaml['out_neurons']},
                'out_activation': {'values': dict_yaml['out_activation']},
                'use_adaptive_clipping': {'values': dict_yaml['use_adaptive_clipping']},
                'use_warmup': {'values': dict_yaml['use_warmup']}

            }
        config.update({'parameters':parameters})

    return config

def hash_string(input_string):
    return hashlib.sha256(input_string.encode()).hexdigest()


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

        name_str = f'{wandb.config.architecture}_{wandb.config.activation}_{wandb.config.dropout}_{wandb.config.use_batchnorm_padding_before_conv1}_{wandb.config.use_padding_pooling_after_conv1}_ \
                        {wandb.config.use_padding_pooling_after_conv1}_{wandb.config.stochastic_depth}_{wandb.config.kernel_sizes}_{wandb.config.strides}_{wandb.config.use_bias}_ \
                        {wandb.config.model_width}_{wandb.config.data}_{wandb.config.lr}_{wandb.config.optimiser}_{wandb.config.weight_decay}_{wandb.config.batch_size}_{wandb.config.loss}_{wandb.config.scheduler}_ \
                        {wandb.config.use_ema}_{wandb.config.apply_label_smoothing}_{wandb.config.label_smoothing_factor}_{wandb.config.rand_augment}_{wandb.config.aug_function}_{wandb.config.use_adaptive_clipping}_{wandb.config.use_warmup}'
            

        encoded = hash_string(name_str)
        train_path = dict(zip(pipeline_parameters['data_types'],pipeline_parameters['train_X_path']))
        val_path = dict(zip(pipeline_parameters['data_types'],pipeline_parameters['val_X_path']))
        print('awdwad')

        #create a run directory if doesn't exist
        os.makedirs(os.path.join(pipeline_parameters['save_dir'],pipeline_parameters['name']), exist_ok=True) 

        #create a run dir for a local save
        os.makedirs(os.path.join(pipeline_parameters['save_dir'],pipeline_parameters['name'],encoded), exist_ok=True)

        #load data
        print('Loading data ...')
        print('\t loading X_train')
        X_train = np.load(os.path.join(pipeline_parameters['main_path'],train_path[wandb.config.data])).astype(np.float16)
        print('\t loading X_val')
        X_val = np.load(os.path.join(pipeline_parameters['main_path'],val_path[wandb.config.data])).astype(np.float16)
        #print('\t loading X_test')
        #X_test = np.load(pipeline_parameters['test_X_path']).astype(np.float16)

        print('\t loading Y_train')
        Y_train = np.load(pipeline_parameters['train_Y_path']).astype(np.float16)
        print('\t loading Y_val')
        Y_val = np.load(pipeline_parameters['val_Y_path']).astype(np.float16)
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
            X_val = np.swapaxes(X_val, -2, -1)


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


        #log the model variables
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
            'use_ema': wandb.config.use_ema,
            'apply_label_smoothing': wandb.config.apply_label_smoothing,
            'label_smoothing_factor': wandb.config.label_smoothing_factor,
            'rand_augment': wandb.config.rand_augment,
            'aug_function': wandb.config.aug_function,
            'use_adaptive_clipping': wandb.config.use_adaptive_clipping,
            'use_warmup': wandb.config.use_warmup
            }
        

        wandb.log(log_dict)

        if not pipeline_parameters['use_pretrained']:
            model = load_model(wandb.config, 'resnet')

        else:
            model = torch.load('/media/data1/anolin/best_model/p_715.365665435791_asymmetric_loss_5.h5') #if on 229
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
                'num_workers': 28,
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
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=8000)

        if wandb.config.scheduler == 'lambda':
            lambda1 = lambda epoch: 0.65 ** wandb.config.num_max_epochs
            scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda1)

        if wandb.config.scheduler == 'triangular2':   
            scheduler = torch.optim.lr_scheduler.CyclicLR(opt, base_lr=wandb.config.lr, max_lr=0.01,step_size_up=4000,mode="triangular2")

        if wandb.config.scheduler == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(opt, gamma=0.7, step_size=2000)

        if wandb.config.use_warmup == True:
            warmup_scheduler = warmup.ExponentialWarmup(opt,warmup_period=1000)

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

        best_loss = np.inf
        list_val_loss = list()
        patience = 2

        try:

            for epoch in range(wandb.config.num_max_epochs):
            
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
                    if wandb.config.use_adaptive_clipping:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)

                    scaler.step(opt)
                    scaler.update()

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

                    del loss
                    del output
                    del data
                    del label

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
                            if wandb.config.scheduler != 'by_plateau':
                                scheduler.step()

                    else:
                        if wandb.config.scheduler != 'by_plateau':
                            scheduler.step()

        

                del train_steps_loss
                del train_steps_MultilabelAccuracy_micro_group
                del train_steps_MultilabelAccuracy_macro_group
                del train_steps_MultilabelAUROC_micro_group
                del train_steps_MultilabelAUROC_macro_group
                del train_steps_MultilabelAUPRC_macro

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

            
                    if bool(pipeline_parameters['log_wandb_performance']):

                        #logg simple stuff
                        log_dict = {
                        "val_loss": mean(val_steps_loss),
                        "val_MultilabelAccuracy_micro":mean(val_steps_MultilabelAccuracy_micro_group),
                        "val_MultilabelAccuracy_macro":mean(val_steps_MultilabelAccuracy_macro_group),
                        "val_MultilabelAUROC_micro":mean(val_steps_MultilabelAUROC_micro_group),
                        "val_MultilabelAUROC_macro":mean(val_steps_MultilabelAUROC_macro_group),
                        "val_MultilabelAUPRC_macro":mean(val_steps_MultilabelAUPRC_macro),
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
                    beat_val_loss = mean(val_steps_loss)

                else:
                    print(f'Not saving model {mean(val_steps_loss)} not better than {beat_val_loss}')


                if mean(val_steps_MultilabelAUPRC_macro) < 0.1:
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
                    gc.collect()
                    torch.cuda.empty_cache()
                    return 0

                if mean(val_steps_MultilabelAUPRC_macro) < 0.2 and epoch+1 >= 7 and wandb.config.use_warmup == False:
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
                    gc.collect()
                    torch.cuda.empty_cache()
                    return 0

                if mean(val_steps_MultilabelAUPRC_macro) < 0.3 and epoch+1 >= 10:
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
                    gc.collect()
                    torch.cuda.empty_cache()
                    return 0

                elif mean(val_steps_MultilabelAUPRC_macro) < 0.40 and epoch+1 >= 15 :
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
            gc.collect()
            torch.cuda.empty_cache()
            return 0

    time.sleep(60)
    del X_train
    del Y_train
    del X_val
    del Y_val
    gc.collect()
    torch.cuda.empty_cache()


# [37, 7, 7, 5, 5]
# [3, 3, 3, 2, 2]

# select a model to run
sweep_configuration = init_wandb()
pp.pprint(sweep_configuration)
sweep_id = wandb.sweep(sweep=sweep_configuration, project='benchmark_resnet_2', entity='mhi_ai') #entity='mhi_ai',
wandb.agent(sweep_id, function=functools.partial(main), count=100)
