
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
from torcheval.metrics import BinaryAUROC,BinaryAccuracy, BinaryF1Score, BinaryPrecision, BinaryRecall
from statistics import mean
import torch.optim as optim
import pytorch_warmup as warmup
import gc
import wandb
import functools
import hashlib
import time
from ResNet import *

# from vit_pytorch import *
# from cross_vit_pytorch import *
# from EffcientNet import *

# from validation_of_datasets import *

from utils import *
import glob
from autoclip.torch import QuantileClip
from termcolor import colored  

from utils import FocalLoss_tahsin
import warnings

warnings.filterwarnings("ignore") #this is for some transformation method
#warnings.filterwarnings("ignore", category=RuntimeWarning) 
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:764"

name= "resnet_pytorch_march17"
MODEL_NAME = 'resnet18_pytorch'
model_to_run = 'resnet18_pytorch'
output_dir='/volume/DeepACS/march13/automatic-ecg-diagnosis/exp6-2_data26k'
condition_name='lr_plateau_epoch50_batch126_focal4_0.1'
clean_dir=True
save_dir= '/volume/DeepACS_runs'

#general optimisation parameters
lr= 0.001
scheduler='by_plateau'
optimiser='Adam'
activation= 'relu'
num_max_epochs= 25
ema_value= [0.9999999, 0.80]
apply_label_smoothing= [False, True]
label_smoothing_factor= [0.000001,0.0] #
#rand_augment: [False] #not implemented yet, keep to false
stochastic_depth= [0.5,0.0]
weight_decay= [0.01, 0.0000001] #[0.1,0.0000001]
dropout= [0.0, 0.5]
batch_size= 128
out_neurons=1
out_activation= 'sigmoid'
use_adaptive_clipping= True
use_warmup= True
loss= 'focal_loss_binary'
log_wandb_performance= True



# Overwrite the random run names chosen by wandb

model = ResNet1D(
        input_channels=12,
        architecture='resnet18_ECG',
        activation='relu',
        dropout=0.8,
        use_batchnorm_padding_before_conv1=True,
        use_padding_pooling_after_conv1=True,
        stochastic_depth=0.0,
        kernel_sizes=[16, 16, 16, 16, 16],  # Custom kernel sizes for each stage
        strides=[1, 1, 1, 1, 1],  # Custom strides for each stage
        use_bias=False,
        use_2d = False,
        out_neurons=1,)
 
print(model)



with wandb.init() as run:
    #overwrite the random run names chosen by wandb
    project= "DeepACS"
    run.name = f"{model_to_run}_{condition_name}"
    wandb.config.update({
        "MODEL_NAME": MODEL_NAME,
        "output_dir": output_dir,
        "condition_name": condition_name,
        "lr": lr,
        "scheduler": scheduler,
        "optimiser": optimiser,
        "activation": activation,
        "num_max_epochs": num_max_epochs,
        "ema_value": ema_value[0],
        "apply_label_smoothing": apply_label_smoothing,
        "label_smoothing_factor": label_smoothing_factor,
        "stochastic_depth": stochastic_depth,
        "weight_decay": weight_decay,
        "dropout": dropout,
        "batch_size": batch_size,
        "out_neurons": out_neurons,
        "out_activation": out_activation,
        "use_adaptive_clipping": use_adaptive_clipping,
        "use_warmup": use_warmup,
        "loss": loss,
        "activation": activation
        
        
    })
    
log_dict = {
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

wandb.log(log_dict)
if wandb.config.optimiser == 'Adam':
    opt = optim.Adam(model.parameters(), lr=wandb.config.lr, weight_decay=wandb.config.weight_decay)
    
 
if wandb.config.loss == 'focal_loss_binary':
    criterion =FocalLoss(gamma=3)
    
    
train_params = {'batch_size': wandb.config.batch_size,
                'shuffle': True,
                'num_workers': 18,
                'pin_memory':True}
val_params = {'batch_size': wandb.config.batch_size,
                'shuffle': False,
                'num_workers': 18,
                'pin_memory':True}


DEVICE = 2
set_seed=42
torch.manual_seed(set_seed)

X_train=np.load('/volume/26k/exp6-2_data26k/X_train_scale.npy')
Y_train=np.load('/volume/26k/exp6-3_data26k/Y_train.npy')
X_val=np.load('/volume/26k/exp6-2_data26k/X_val_scale.npy')
Y_val=np.load('/volume/26k/exp6-3_data26k/Y_val.npy')

training_set = ECGDataset(X_train, Y_train)
training_generator = torch.utils.data.DataLoader(training_set, **train_params)

eval_set = ECGDataset(X_val, Y_val)
eval_generator = torch.utils.data.DataLoader(eval_set, **val_params)




# Define metrics AUC, Accuracy, Precision for binary classification



if wandb.config.scheduler == 'by_plateau':
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.1, patience=1, verbose=True)
    
if wandb.config.use_warmup == True:
    warmup_scheduler = warmup.ExponentialWarmup(opt,warmup_period=500)

#use for float16 encoding
scaler = torch.cuda.amp.GradScaler()
print_steps = int(500)
log_wandb_performance: True


# label_names = new_label_names

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

#print wandb.config

from torcheval.metrics import BinaryBalancedAccuracy, BinaryPrecision, BinaryRecall, AUROC, RECALL

print(wandb.config)
accuracy = BinaryBalancedAccuracy().to(DEVICE)
auroc = AUROC(pos_label=1 ).to(DEVICE)
precision = BinaryPrecision(pos_label=1).to(DEVICE)
recall = RECALL(pos_label=1).to(DEVICE)
balanced_accuracy = BinaryBalancedAccuracy().to(DEVICE)


try:
    print("Starting training loop")
    for epoch in range(wandb.config.num_max_epochs):
        model.train()
        train_steps_loss = list()
        train_steps_loss = list()
        train_steps_accuracy = list()
        train_steps_auroc = list()
        train_steps_balanced_accuracy = list()
        train_steps_precision = list()
        train_steps_recall = list()

        
                #step accumulators 
        for iter_, (data, label) in enumerate(tqdm(training_generator)):
            data = data.to(DEVICE)
            print(f"data shape {data.shape}")

            label = label.type(torch.long).to(DEVICE)
            print(f"label shape {label.shape}")

            
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                output = model(data)
                
                output = torch.nan_to_num(output)   
                print(f"output shape {output.shape}")
                
                loss = criterion(output, label.float())
                print(f"loss shape {loss.shape}")

            opt.zero_grad()
            scaler.scale(loss).backward()

            scaler.unscale_(opt)
            
            scaler.step(opt)
            scaler.update()
            ema_updater.update()
            #acquire train metrics
            train_steps_loss.append(loss.item())
            train_steps_accuracy.append(accuracy(output, label).item())
            train_steps_auroc.append(auroc(output, label).item())
            train_steps_balanced_accuracy.append(balanced_accuracy(output, label).item())
            train_steps_precision.append(precision(output, label).item())
            train_steps_recall.append(recall(output, label).item())
            
            
            current_lr = opt.param_groups[0]['lr']

            if iter_ % print_steps == 0 and iter_ != 0:
                print(f"Train E:{epoch+1} Step:{iter_} LR {current_lr} Loss {wandb.config.loss}: {'{:.3f}'.format(mean(train_steps_loss[iter_-print_steps:iter_]))} Acc {'{:.3f}'.format(mean(train_steps_accuracy[iter_-print_steps:iter_]))} ROC {'{:.3f}'.format(mean(train_steps_auroc[iter_-print_steps:iter_]))} Precision {'{:.3f}'.format(mean(train_steps_precision[iter_-print_steps:iter_]))} Recall {'{:.3f}'.format(mean(train_steps_recall[iter_-print_steps:iter_]))} Balanced Accuracy {'{:.3f}'.format(mean(train_steps_balanced_accuracy[iter_-print_steps:iter_]))}")

            if bool(log_wandb_performance):
                # Log metrics to wandb
                wandb.log({
                    "Train Loss": mean(train_steps_loss),
                    "Train Accuracy": mean(train_steps_accuracy),
                    "Train AUROC": mean(train_steps_auroc),
                    "Train Precision": mean(train_steps_precision),
                    "Train Recall": mean(train_steps_recall),
                    "Train Balanced Accuracy": mean(train_steps_balanced_accuracy),
                })

            if wandb.config.use_warmup:
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
        del train_steps_accuracy
        del train_steps_auroc
        del train_steps_precision
        del train_steps_recall
        del train_steps_balanced_accuracy
        
        model.eval()
        original_state_dict = ema_updater.apply_shadow()
        with torch.no_grad():
            val_steps_loss = list()
            val_steps_accuracy = list()
            val_steps_auroc = list()
            val_steps_precision = list()
            val_steps_recall = list()
            val_steps_balanced_accuracy = list()

            for iter_, (data, label) in enumerate(tqdm(eval_generator)):
                data = data.to(DEVICE)
                label = label.type(torch.long).to(DEVICE)
                if wandb.config.loss not in ['focal_loss_binary']:
                    label = label.type(torch.int32).to(DEVICE)
                else:
                    label = label.type(torch.long).to(DEVICE)

                output = model(data)
                output = torch.nan_to_num(output)
                loss = criterion(output, label.float())

                val_steps_loss.append(loss.item())
                val_steps_accuracy.append(accuracy(output, label).item())
                val_steps_auroc.append(auroc(output, label).item())
                val_steps_precision.append(precision(output, label).item())
                val_steps_recall.append(recall(output, label).item())
                val_steps_balanced_accuracy.append(balanced_accuracy(output, label).item())

            if bool(log_wandb_performance):
                # Log metrics to wandb
                wandb.log({
                    "Val Loss": mean(val_steps_loss),
                    "Val Accuracy": mean(val_steps_accuracy),
                    "Val AUROC": mean(val_steps_auroc),
                    "Val Precision": mean(val_steps_precision),
                    "Val Recall": mean(val_steps_recall),
                    "Val Balanced Accuracy": mean(val_steps_balanced_accuracy),
                })
            
            print(f"Validation E:{epoch+1} Loss {wandb.config.loss}: {'{:.3f}'.format(mean(val_steps_loss))} Acc {'{:.3f}'.format(mean(val_steps_accuracy))} ROC {'{:.3f}'.format(mean(val_steps_auroc))} Precision {'{:.3f}'.format(mean(val_steps_precision))} Recall {'{:.3f}'.format(mean(val_steps_recall))} Balanced Accuracy {'{:.3f}'.format(mean(val_steps_balanced_accuracy))}")
            
            ema_updater.restore_original(original_state_dict)
            val_steps_loss_ = mean(val_steps_loss)
            list_val_loss.append(val_steps_loss_)
            
            
            if val_steps_loss_ < best_loss:
                best_loss = val_steps_loss_
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f"Stopping early at epoch {epoch}.")
                break
                
                
                
        
            #save model if improvement
            if mean(val_steps_loss) < beat_val_loss:
                print(f'Saving model {mean(val_steps_loss)} bettter than {beat_val_loss}')
                if clean_dir:
                    for f in glob.glob(os.path.join(save_dir,name,'*')):
                        try:
                            if os.path.isfile(f):  # Ensure it's a file, not a directory
                                os.remove(f)
                        except:
                            pass


                    torch.save(model, os.path.join(save_dir,name,f'{model_to_run}_{mean(val_steps_loss)}_{wandb.config.loss}_{epoch}.h5'))
                    
                # reset the patience counter
            # del val_steps_loss
            # del val_steps_accuracy
            # del val_steps_auroc
            # del val_steps_precision
            # del val_steps_recall
            # del val_steps_balanced_accuracy

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
            print(0)
    
del loss
del output
del data
del label

del train_steps_loss
del train_steps_accuracy
del train_steps_auroc
del train_steps_precision
del train_steps_recall
del train_steps_balanced_accuracy