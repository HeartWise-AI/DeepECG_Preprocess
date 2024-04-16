import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import heartpy as hp
import random
import math
import sys
import heartpy as hp
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from random import shuffle
from scipy.interpolate import CubicSpline
from numba import jit
from tqdm.contrib import tzip
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from numpy.lib.format import read_magic, _read_array_header

###################
# Label smoothing #
###################

def smooth_labels(labels, smoothing=0.1):
    """
    Apply label smoothing.
    :param labels: Original labels.
    :param smoothing: Label smoothing factor.
    """
    with torch.no_grad():
        num_classes = labels.shape[1]
        labels = labels * (1 - smoothing) + (smoothing / num_classes)
    return labels


########
# LOSS #
########

# multilabel

class Hill(nn.Module):
    r""" Hill as described in the paper "Robust Loss Design for Multi-Label Learning with Missing Labels "

    .. math::
        Loss = y \times (1-p_{m})^\gamma\log(p_{m}) + (1-y) \times -(\lambda-p){p}^2 

    where : math:`\lambda-p` is the weighting term to down-weight the loss for possibly false negatives,
          : math:`m` is a margin parameter, 
          : math:`\gamma` is a commonly used value same as Focal loss.

    .. note::
        Sigmoid will be done in loss. 

    Args:
        lambda (float): Specifies the down-weight term. Default: 1.5. (We did not change the value of lambda in our experiment.)
        margin (float): Margin value. Default: 1 . (Margin value is recommended in [0.5,1.0], and different margins have little effect on the result.)
        gamma (float): Commonly used value same as Focal loss. Default: 2

    """

    def __init__(self, lamb: float = 1.5, margin: float = 1.0, gamma: float = 2.0,  reduction: str = 'sum') -> None:
        super(Hill, self).__init__()
        self.lamb = lamb
        self.margin = margin
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        call function as forward

        Args:
            logits : The predicted logits before sigmoid with shape of :math:`(N, C)`
            targets : Multi-label binarized vector with shape of :math:`(N, C)`

        Returns:
            torch.Tensor: loss
        """

        # Calculating predicted probability
        logits_margin = logits - self.margin
        pred_pos = torch.sigmoid(logits_margin)
        pred_neg = torch.sigmoid(logits)

        # Focal margin for postive loss
        pt = (1 - pred_pos) * targets + (1 - targets)
        focal_weight = pt ** self.gamma

        # Hill loss calculation
        los_pos = targets * torch.log(pred_pos)
        los_neg = (1-targets) * -(self.lamb - pred_neg) * pred_neg ** 2

        loss = -(los_pos + los_neg)
        loss *= focal_weight

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

class SPLC(nn.Module):
    r""" SPLC loss as described in the paper "Simple Loss Design for Multi-Label Learning with Missing Labels "

    .. math::
        &L_{SPLC}^+ = loss^+(p)
        &L_{SPLC}^- = \mathbb{I}(p\leq \tau)loss^-(p) + (1-\mathbb{I}(p\leq \tau))loss^+(p)

    where :math:'\tau' is a threshold to identify missing label 
          :math:`$\mathbb{I}(\cdot)\in\{0,1\}$` is the indicator function, 
          :math: $loss^+(\cdot), loss^-(\cdot)$ refer to loss functions for positives and negatives, respectively.

    .. note::
        SPLC can be combinded with various multi-label loss functions. 
        SPLC performs best combined with Focal margin loss in our paper. Code of SPLC with Focal margin loss is released here.
        Since the first epoch can recall few missing labels with high precision, SPLC can be used ater the first epoch.
        Sigmoid will be done in loss. 

    Args:
        tau (float): threshold value. Default: 0.6
        change_epoch (int): which epoch to combine SPLC. Default: 1
        margin (float): Margin value. Default: 1
        gamma (float): Hard mining value. Default: 2
        reduction (string, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Default: ``'sum'``

        """

    def __init__(self,
                 tau: float = 0.6,
                 change_epoch: int = 1,
                 margin: float = 1.0,
                 gamma: float = 2.0,
                 reduction: str = 'sum') -> None:
        super(SPLC, self).__init__()
        self.tau = tau
        self.change_epoch = change_epoch
        self.margin = margin
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.LongTensor,
                epoch) -> torch.Tensor:
        """
        call function as forward

        Args:
            logits : The predicted logits before sigmoid with shape of :math:`(N, C)`
            targets : Multi-label binarized vector with shape of :math:`(N, C)`
            epoch : The epoch of current training.

        Returns:
            torch.Tensor: loss
        """
        # Subtract margin for positive logits
        logits = torch.where(targets == 1, logits-self.margin, logits)
        
        # SPLC missing label correction
        if epoch >= self.change_epoch:
            targets = torch.where(
                torch.sigmoid(logits) > self.tau,
                torch.tensor(1).cuda(), targets)
        
        pred = torch.sigmoid(logits)

        # Focal margin for postive loss
        pt = (1 - pred) * targets + pred * (1 - targets)
        focal_weight = pt**self.gamma

        los_pos = targets * F.logsigmoid(logits)
        los_neg = (1 - targets) * F.logsigmoid(-logits)

        loss = -(los_pos + los_neg)
        loss *= focal_weight

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=True):
        super(AsymmetricLoss, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            loss *= one_sided_w

        return -loss.sum()

class AsymmetricLossOptimized(nn.Module):
    ''' Notice - optimized version, minimizes memory allocation and gpu uploading,
    favors inplace operations'''

    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=False):
        super(AsymmetricLossOptimized, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

        # prevent memory allocation and gpu uploading every iteration, and encourages inplace operations
        self.targets = self.anti_targets = self.xs_pos = self.xs_neg = self.asymmetric_w = self.loss = None

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        self.targets = y
        self.anti_targets = 1 - y

        # Calculating Probabilities
        self.xs_pos = torch.sigmoid(x)
        self.xs_neg = 1.0 - self.xs_pos

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            self.xs_neg.add_(self.clip).clamp_(max=1)

        # Basic CE calculation
        self.loss = self.targets * torch.log(self.xs_pos.clamp(min=self.eps))
        self.loss.add_(self.anti_targets * torch.log(self.xs_neg.clamp(min=self.eps)))

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            self.xs_pos = self.xs_pos * self.targets
            self.xs_neg = self.xs_neg * self.anti_targets
            self.asymmetric_w = torch.pow(1 - self.xs_pos - self.xs_neg,
                                          self.gamma_pos * self.targets + self.gamma_neg * self.anti_targets)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            self.loss *= self.asymmetric_w

        return -self.loss.sum()

def calculate_class_weights(one_hot_labels):
    """
    Calculate class weights based on the frequency of each class in the one-hot encoded labels.

    Args:
    - one_hot_labels (torch.Tensor): One-hot encoded labels with binary values (0 or 1) and shape (num_samples, num_classes).

    Returns:
    - class_weights (torch.Tensor): Computed class weights.
    """
    num_classes = one_hot_labels.size(1)

    # Count the number of positive occurrences of each class
    class_counts = torch.sum(one_hot_labels, dim=0)

    # Calculate class weights as the inverse of class frequencies
    class_weights = 1.0 / (class_counts + 1e-8)  # Adding a small epsilon to avoid division by zero

    # Normalize the weights to sum to 1
    class_weights /= torch.sum(class_weights)

    return class_weights

class WeightedMultilabelBCELoss(nn.Module):
    def __init__(self, weight=None):
        super(WeightedMultilabelBCELoss, self).__init__()
        self.weight = weight

    def forward(self, input, target):
        # input: raw logits
        # target: binary target labels

        # Apply sigmoid to convert logits to probabilities
        input = torch.sigmoid(input)

        # Calculate the binary cross-entropy loss
        loss = F.binary_cross_entropy(input, target, weight=self.weight)

        return loss

class BCEWithLogitsLossWithLabelSmoothing(torch.nn.Module):
    def __init__(self, label_smoothing=0.0, pos_weight=None, reduction='mean'):
        super(BCEWithLogitsLossWithLabelSmoothing, self).__init__()
        self.label_smoothing = label_smoothing
        self.pos_weight = pos_weight
        self.reduction = reduction

    def forward(self, input, target_smooth):
        # Apply label smoothing, adjusting targets away from 0 and 1 towards 0.5        
        if self.pos_weight is not None:
            # If pos_weight is specified, apply it
            loss = F.binary_cross_entropy_with_logits(input, target_smooth, pos_weight=self.pos_weight, reduction=self.reduction)
        else:
            loss = F.binary_cross_entropy_with_logits(input, target_smooth, reduction=self.reduction)
        
        return loss

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, one_hot_labels):
        """
        Calculate Focal Loss.

        Args:
        - logits (torch.Tensor): Raw logits from the model (shape: [batch_size, num_classes]).
        - one_hot_labels (torch.Tensor): One-hot encoded labels with binary values (0 or 1, shape: [batch_size, num_classes]).

        Returns:
        - loss (torch.Tensor): Computed Focal Loss.
        """
        pt = F.sigmoid(logits) * one_hot_labels + (1 - F.sigmoid(logits)) * (1 - one_hot_labels)
        focal_weight = (self.alpha * one_hot_labels + (1 - self.alpha) * (1 - one_hot_labels)) * torch.pow(1 - pt, self.gamma)

        loss = F.binary_cross_entropy_with_logits(logits, one_hot_labels, reduction='none')
        focal_loss = focal_weight * loss

        if self.reduction == 'mean':
            return torch.mean(focal_loss)
        elif self.reduction == 'sum':
            return torch.sum(focal_loss)
        else:
            return focal_loss

nINF = -100
class TwoWayLoss(nn.Module):

    def __init__(self, Tp=4., Tn=1.):
        super(TwoWayLoss, self).__init__()
        self.Tp = Tp
        self.Tn = Tn

    def forward(self, x, y):
        class_mask = (y > 0).any(dim=0)
        sample_mask = (y > 0).any(dim=1)

        # Calculate hard positive/negative logits
        pmask = y.masked_fill(y <= 0, nINF).masked_fill(y > 0, float(0.0))
        plogit_class = torch.logsumexp(-x/self.Tp + pmask, dim=0).mul(self.Tp)[class_mask]
        plogit_sample = torch.logsumexp(-x/self.Tp + pmask, dim=1).mul(self.Tp)[sample_mask]
    
        nmask = y.masked_fill(y != 0, nINF).masked_fill(y == 0, float(0.0))
        nlogit_class = torch.logsumexp(x/self.Tn + nmask, dim=0).mul(self.Tn)[class_mask]
        nlogit_sample = torch.logsumexp(x/self.Tn + nmask, dim=1).mul(self.Tn)[sample_mask]

        return torch.nn.functional.softplus(nlogit_class + plogit_class).mean() + \
                torch.nn.functional.softplus(nlogit_sample + plogit_sample).mean()

class BinaryFocalLossWithLogits(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        """
        Binary Focal Loss with Logits

        Parameters:
        alpha (float): Weighting factor for the rare class. Defaults to 0.25.
        gamma (float): Focusing parameter to minimize the loss contribution from easy examples and extend focus on hard examples. Defaults to 2.0.
        reduction (str): Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'. 'none': no reduction will be applied, 'mean': the sum of the output will be divided by the number of elements in the output, 'sum': the output will be summed.
        """
        super(BinaryFocalLossWithLogits, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Compute the focal loss given inputs (logits) and targets (true labels)

        Parameters:
        inputs (tensor): Logits at the model's output.
        targets (tensor): Target values, should be same size as inputs.

        Returns:
        tensor: Computed focal loss.
        """
        # Compute the binary cross entropy loss
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')

        # Compute probabilities from logits
        probs = torch.sigmoid(inputs)

        # Compute the focal loss component
        focal_loss = self.alpha * (1 - probs) ** self.gamma * bce_loss

        if self.reduction == 'mean':
            return torch.mean(focal_loss)
        elif self.reduction == 'sum':
            return torch.sum(focal_loss)
        else:
            return focal_loss


#multiclass

class FocalLossMulticlass(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = torch.softmax(inputs, dim=1)
        targets = F.one_hot(targets, num_classes=inputs.shape[1]).permute(0, 3, 1, 2).float()
        
        intersection = (inputs * targets).sum(dim=(2, 3))
        union = inputs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
        
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1 - dice.mean()
        return dice_loss

class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, smooth=1.):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = torch.softmax(inputs, dim=1)
        targets = F.one_hot(targets, num_classes=inputs.shape[1]).permute(0, 3, 1, 2).float()
        
        TP = (inputs * targets).sum(dim=(2, 3))
        FP = ((1 - targets) * inputs).sum(dim=(2, 3))
        FN = (targets * (1 - inputs)).sum(dim=(2, 3))
        
        Tversky = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
        Tversky_loss = 1 - Tversky.mean()
        return Tversky_loss

class FocalCosineLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=1.0, reduction='mean'):
        super(FocalCosineLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, input, target):
        # Assuming input is already logits and target is one-hot encoded
        cosine_loss = F.cosine_embedding_loss(input, F.one_hot(target, num_classes=input.size(-1)),
                                              torch.ones(input.size(0)).to(input.device), reduction=self.reduction)
        pt = 1 - cosine_loss
        focal_loss = self.alpha * (1 - pt)**self.gamma * cosine_loss

        return focal_loss


#########################
# Dynamic tiime warping #
#########################
    
# heavily modified from https://github.com/uchidalab/time_series_augmentation

RETURN_VALUE = 0
RETURN_PATH = 1
RETURN_ALL = -1

# Core DTW
def _traceback(DTW, slope_constraint):
    i, j = np.array(DTW.shape) - 2
    max_len = i + j
    p = np.zeros(max_len, dtype=np.int32)
    q = np.zeros(max_len, dtype=np.int32)
    idx = 0

    while i > 0 and j > 0:
        if slope_constraint == "asymmetric":
            tb = np.argmin((DTW[i-1, j], DTW[i-1, j-1], DTW[i-1, max(j-2, 0)]))
            if tb == 0:
                i -= 1
            elif tb == 1:
                i -= 1
                j -= 1
            else:
                i -= 1
                j -= 2
        elif slope_constraint == "symmetric":
            tb = np.argmin((DTW[i-1, j-1], DTW[i-1, j], DTW[i, j-1]))
            if tb == 0:
                i -= 1
                j -= 1
            elif tb == 1:
                i -= 1
            else:
                j -= 1
        else:
            raise ValueError("Unknown slope constraint %s" % slope_constraint)

        p[idx] = i
        q[idx] = j
        idx += 1

    return p[:idx][::-1], q[:idx][::-1]

def dtw(prototype, sample, return_flag=RETURN_VALUE, slope_constraint="asymmetric", window=None):
    p = prototype.shape[0]
    s = sample.shape[0]

    assert p != 0, "Prototype empty!"
    assert s != 0, "Sample empty!"
    
    if window is None:
        window = s

    # Efficient calculation of the cost matrix
    cost = np.full((p, s), np.inf)
    for i in range(p):
        start = max(0, i - window)
        end = min(s, i + window) + 1
        cost[i, start:end] = np.linalg.norm(sample[start:end] - prototype[i], axis=1)

    DTW = _cummulative_matrix(cost, slope_constraint, window)
        
    if return_flag == RETURN_ALL:
        return DTW[-1, -1], cost, DTW[1:, 1:], _traceback(DTW, slope_constraint)
    elif return_flag == RETURN_PATH:
        return _traceback(DTW, slope_constraint)
    else:
        return DTW[-1, -1]

@jit(nopython=True)
def _cummulative_matrix(cost, slope_constraint, window):
    p, s = cost.shape
    DTW = np.full((p + 1, s + 1), np.inf)
    DTW[0, 0] = 0.0

    if slope_constraint == "asymmetric":
        for i in range(1, p + 1):
            start = max(2, i - window)
            end = min(s + 1, i + window + 1)
            for j in range(start, end):
                # Ensure j-2 does not go out of bounds
                j_2 = max(j - 2, 0)
                DTW[i, j] = cost[i - 1, j - 1] + min(DTW[i - 1, j - 1], DTW[i - 1, j], DTW[i - 1, j_2])

            # Special handling for the start of each row
            if i <= window + 1:
                DTW[i, 1] = cost[i - 1, 0] + min(DTW[i - 1, 0], DTW[i - 1, 1])

    elif slope_constraint == "symmetric":
        for i in range(1, p + 1):
            start = max(1, i - window)
            end = min(s + 1, i + window + 1)
            for j in range(start, end):
                DTW[i, j] = cost[i - 1, j - 1] + min(DTW[i - 1, j - 1], DTW[i, j - 1], DTW[i - 1, j])

    else:
        raise ValueError("Unknown slope constraint: " + slope_constraint)

    return DTW

@jit(nopython=True)
def calculate_cost_matrix(prototype_pad, sample_pad, p, s, p_feature_len, s_feature_len, window):
    cost = np.full((p, s), np.inf)
    for i in range(p):
        start = max(0, i - window)
        end = min(s, i + window + 1)
        for j in range(start, end):
            prototype_slice = prototype_pad[i:i + p_feature_len]
            sample_slice = sample_pad[j:j + s_feature_len]
            if prototype_slice.shape[0] == p_feature_len and sample_slice.shape[0] == s_feature_len:
                cost[i, j] = np.sum((sample_slice - prototype_slice) ** 2)
    return cost

def shape_dtw(prototype, sample, return_flag=RETURN_VALUE, slope_constraint="asymmetric", window=None, descr_ratio=0.05):
    p = prototype.shape[0]
    s = sample.shape[0]
    if p == 0 or s == 0:
        return None  # Or other appropriate return value for empty input

    if window is None:
        window = s

    p_feature_len = np.clip(np.round(p * descr_ratio), 5, 100).astype(int)
    s_feature_len = np.clip(np.round(s * descr_ratio), 5, 100).astype(int)
    half_p_feature_len = p_feature_len // 2
    half_s_feature_len = s_feature_len // 2

    prototype_pad = np.pad(prototype, ((half_p_feature_len, p_feature_len - half_p_feature_len), (0, 0)), mode="edge") 
    sample_pad = np.pad(sample, ((half_s_feature_len, s_feature_len - half_s_feature_len), (0, 0)), mode="edge") 

    cost = calculate_cost_matrix(prototype_pad, sample_pad, p, s, p_feature_len, s_feature_len, window)

    DTW = _cummulative_matrix(cost, slope_constraint=slope_constraint, window=window)

    if return_flag == RETURN_ALL:
        return DTW[-1,-1], cost, DTW[1:,1:], _traceback(DTW, slope_constraint)
    elif return_flag == RETURN_PATH:
        return _traceback(DTW, slope_constraint)
    else:
        return DTW[-1,-1]

##################
# Tranformations #
##################
    
# modified from https://github.com/uchidalab/time_series_augmentation
# modified for speed and consistency of the transformation for ECGs/along channels
    
#jitter
def jitter(x, sigma=0.03):
    #print(x.shape)
    # https://arxiv.org/pdf/1706.00527.pdf
    a,b,_ = x.shape  #assuming dim [N,2500,12]
    return x + np.repeat(np.random.normal(loc=0., scale=sigma, size=[a,b]).astype(np.float16)[:, :, np.newaxis], 12, axis=-1)

#scaling
def scaling(x, sigma=0.1):
    a,_,_ = x.shape  #assuming dim [N,2500,12]
    factor = np.random.normal(loc=1., scale=sigma, size=a).astype(np.float16)
    factor = np.repeat(factor[:,np.newaxis,np.newaxis], 12, axis=-1)
    return np.multiply(x, factor)

# permutate beats aside from the partial ones at the extreme of the waveform
def process_slice(a_):
    try:
        wd, m = hp.process(a_, 250) #use the optimised heartpy beatfinder
        return wd['peaklist']
    except:
        return []

def beat_permutation(x):

    num_threads = 4
    results = []
    #get the R peaks
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
    # Map the process_slice function to each slice of the array
        for result in executor.map(process_slice, x[:, :, 0]): #find the beat in lead 1
            results.append(result)

    for pos,cuts in enumerate(results): #shuffle the blocks
        if len(cuts) > 3:
            try:
                slices = [x[pos, cuts[j]:cuts[j+1],:] for j in range(len(cuts)-1)]
                shuffle(slices) 
                slices = [x[pos, 0:cuts[0],:]] + slices + [x[pos, cuts[-1]:2500,:]]
                x[pos] = np.concatenate(slices, axis=0)
            except:
                pass
        else:
            pass
    return x


#magnitude warp, basically move the ECGs magnitude using a cubic spline with 4 knots
def single_instance_warp(i, x_slice, warp_steps, random_warps, T, C):
    """
    Apply warping to a single instance (slice) of the array.
    """
    warper = CubicSpline(warp_steps, random_warps[i])(np.arange(T))
    warped_slice = np.zeros_like(x_slice)
    for c in range(C):
        warped_slice[:, c] = x_slice[:, c] * warper
    return warped_slice

def magnitude_warp_uniform_multithreaded(x, sigma=0.2, knot=4, num_threads=24):
    """
    Multithreaded function to apply magnitude warping uniformly across channels.
    """
    N, T, C = x.shape
    warp_steps = np.linspace(0, T-1., num=knot+2)
    random_warps = np.random.normal(loc=1.0, scale=sigma, size=(N, knot+2))

    ret = np.zeros_like(x)

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        # Submit tasks for each instance in N
        futures = [executor.submit(single_instance_warp, i, x[i], warp_steps, random_warps, T, C) for i in range(N)]

        # Collect and assign results while maintaining order
        for i, future in enumerate(futures):
            ret[i] = future.result()

    return ret


#time warp is analogous to the magnitude warp simply the cubic spline is applied to the time dimensiob
def single_instance_time_warp(i, x_slice, warp_steps, random_warps, T, C):
    """
    Apply time warping to a single instance (slice) of the array.
    """
    warped_slice = np.zeros_like(x_slice)
    for dim in range(C):
        time_warp = CubicSpline(warp_steps[:,dim], warp_steps[:,dim] * random_warps[i,:,dim])(np.arange(T))
        scale = (T-1)/time_warp[-1]
        warped_slice[:,dim] = np.interp(np.arange(T), np.clip(scale*time_warp, 0, T-1), x_slice[:,dim]).T
    return warped_slice

def time_warp_multithreaded(x, sigma=0.2, knot=4, num_threads=24):
    """
    Multithreaded function to apply time warping.
    """
    N, T, C = x.shape
    warp_steps = (np.ones((C,1)) * np.linspace(0, T-1., num=knot+2)).T
    random_warps = np.random.normal(loc=1.0, scale=sigma, size=(N, knot+2, C))

    ret = np.zeros_like(x)

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        # Submit tasks for each instance in N
        futures = [executor.submit(single_instance_time_warp, i, x[i], warp_steps, random_warps, T, C) for i in range(N)]

        # Collect and assign results while maintaining order
        for i, future in enumerate(futures):
            ret[i] = future.result()

    return ret


#window slicing here takes 90 of the waveform and extends it to fit the original size
def single_instance_window_slice(x_slice, target_len, start, end, T, C):
    """
    Apply window slicing to a single instance (slice) of the array.
    """
    warped_slice = np.zeros_like(x_slice)
    for dim in range(C):
        warped_slice[:, dim] = np.interp(np.linspace(0, target_len, num=T), np.arange(target_len), x_slice[start:end, dim]).T
    return warped_slice

def window_slice_multithreaded(x, reduce_ratio=0.90, num_threads=24):
    """
    Multithreaded function to apply window slicing uniformly across channels.
    """
    N, T, C = x.shape
    target_len = np.ceil(reduce_ratio * T).astype(int)
    if target_len >= T:
        return x

    starts = np.random.randint(low=0, high=T - target_len, size=(N)).astype(int)
    ends = (target_len + starts).astype(int)

    ret = np.zeros_like(x)

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        # Submit tasks for each instance in N
        futures = [executor.submit(single_instance_window_slice, x[i], target_len, starts[i], ends[i], T, C) for i in range(N)]

        # Collect and assign results while maintaining order
        for i, future in enumerate(futures):
            ret[i] = future.result()

    return ret


# window warp take 10% of the waveform and randomly slows it by 50% or increases its speed by 200%
def single_instance_window_warp(x_slice, warp_scale, window_start, window_end, window_steps, T, C):
    """
    Apply window warping to a single instance (slice) of the array.
    """
    warped_slice = np.zeros_like(x_slice)
    for dim in range(C):
        start_seg = x_slice[:window_start, dim]
        window_seg = np.interp(np.linspace(0, window_end - window_start - 1, num=int((window_end - window_start) * warp_scale)), window_steps, x_slice[window_start:window_end, dim])
        end_seg = x_slice[window_end:, dim]
        warped = np.concatenate((start_seg, window_seg, end_seg))
        warped_slice[:, dim] = np.interp(np.arange(T), np.linspace(0, T - 1, num=warped.size), warped).T
    return warped_slice


def window_warp_multithreaded(x, window_ratio=0.1, scales=[0.5, 2.], num_threads=48):
    """
    Multithreaded function to apply window warping uniformly across channels.
    """
    N, T, C = x.shape
    warp_scales = np.random.choice(scales, N)
    warp_size = np.ceil(window_ratio * T).astype(int)
    window_steps = np.arange(warp_size)

    window_starts = np.random.randint(low=1, high=T - warp_size - 1, size=N).astype(int)
    window_ends = (window_starts + warp_size).astype(int)

    ret = np.zeros_like(x)

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        # Submit tasks for each instance in N
        futures = [executor.submit(single_instance_window_warp, x[i], warp_scales[i], window_starts[i], window_ends[i], window_steps, T, C) for i in range(N)]

        # Collect and assign results while maintaining order
        for i, future in enumerate(futures):
            ret[i] = future.result()

    return ret

#↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑
# methods above are label-less

#↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
#the methods require labels 


#Suboptimal Warped Time Series Generator (SPAWNER)
# takes the DTW of two label-equal samples with random permutations

def jitter_small(x, sigma=0.03):
    # https://arxiv.org/pdf/1706.00527.pdf
    a,_ = x.shape  #assuming dim [N,2500,12]
    return x + np.repeat(np.random.normal(loc=0., scale=sigma, size=[a]).astype(np.float16)[:, np.newaxis], 12, axis=-1)

def spawner_worker(i, pat, random_sample, random_point, window, orig_steps, sigma):

    result = np.zeros((2500,12))
    if random_sample is not None:
        #print("Augmenting")
        path1 = dtw(pat[:random_point], random_sample[:random_point], RETURN_PATH, slope_constraint="symmetric", window=window)
        path2 = dtw(pat[random_point:], random_sample[random_point:], RETURN_PATH, slope_constraint="symmetric", window=window)
        combined = np.concatenate((np.vstack(path1), np.vstack(path2+random_point)), axis=1)
        mean = np.mean([pat[combined[0]], random_sample[combined[1]]], axis=0)
        for dim in range(12):
            result[:, dim] = np.interp(orig_steps, np.linspace(0, 2500-1., num=mean.shape[0]), mean[:, dim])

    else:
        #print("There is only one pattern of class, skipping pattern average")
        result = pat

    return jitter_small(result, sigma=sigma)  # Assuming jitter is another defined function

def remove_matching_indices(X_large, single_example, indices):
    filtered_indices = []

    for idx in indices:
        # Compare the single_example with the specific row in X_large
        if not np.array_equal(X_large[idx], single_example):
            filtered_indices.append(idx)

    return np.array(filtered_indices)

def find_matching_rows(label, main_y):
    # Broadcasting comparison over all rows of main_y
    matches = (main_y == label).all(axis=1)
    return np.where(matches)[0]

def spawner(x, main_x, labels, main_y, sigma=0.05, num_threads=24):
    if main_x is None or main_y is None:
        main_x = x
        main_y = labels

    labels, main_y = labels.astype(int), main_y.astype(int)
    random_points = np.random.randint(low=1, high=x.shape[1]-1, size=x.shape[0])
    window = np.ceil(x.shape[1] / 10.).astype(int)
    orig_steps = np.arange(x.shape[1])

    # Precompute comparison samples
    comparison_samples = []
    for i, label in enumerate(labels):

        choices = find_matching_rows(label,main_y)
        choices = remove_matching_indices(main_x,x[i],choices)

        #choices = choices[choices != i]  # Exclude the current index
        if choices.size > 0:
            random_choice = np.random.choice(choices)
            comparison_samples.append(main_x[random_choice])
        else:
            comparison_samples.append(None)

    ret = np.zeros_like(x)
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(spawner_worker, i, x[i], comparison_samples[i], random_points[i], window, orig_steps, sigma) for i in range(x.shape[0])]
        for i, future in enumerate(futures):
            ret[i] = future.result()

    return ret

#guided warp as proposed by https://arxiv.org/pdf/2004.08780.pdf with significant acceleration
def single_instance_random_guided_warp(i, x_slice, random_prototype_index, full_dataset, slope_constraint, dtw_type, window):
    """
    Apply random guided warp to a single instance using a pre-selected prototype index.
    """
    T, C = x_slice.shape
    orig_steps = np.arange(T)
    warped_instance = np.zeros_like(x_slice)

    if random_prototype_index is not None:
        random_prototype = full_dataset[random_prototype_index]

        # Apply DTW to align the prototype and the current instance
        if dtw_type == "shape":
            random_prototype = random_prototype.astype(np.float32)
            x_slice = x_slice.astype(np.float32)
            path = shape_dtw(random_prototype, x_slice, RETURN_PATH, slope_constraint=slope_constraint, window=window)
        else:
            path = dtw(random_prototype, x_slice, RETURN_PATH, slope_constraint=slope_constraint, window=window)

        # Time warp
        warped = x_slice[path[1]]
        for dim in range(C):
            warped_instance[:, dim] = np.interp(orig_steps, np.linspace(0, T-1, num=warped.shape[0]), warped[:, dim]).T
    else:
        # No other samples from the same class, use the original instance
        warped_instance = x_slice

    return warped_instance

def random_guided_warp_multithreaded(batch, full_dataset, batch_labels, full_dataset_labels, slope_constraint="symmetric", dtw_type="normal", use_window=True, num_threads=18):
    """
    Multithreaded random guided warp function with precomputed random prototypes.
    """
    N, T, C = batch.shape
    window = np.ceil(T / 10).astype(int) if use_window else None

    # Precompute random prototype choices
    random_prototype_indices = []
    for i in range(N):
        current_label = batch_labels[i]
        choices = np.where(full_dataset_labels == current_label)[0]

        
        choices = choices[choices != i]  # Exclude the current instance
        random_prototype_indices.append(None if len(choices) == 0 else choices[-1])

    processed_data = np.zeros_like(batch)

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(single_instance_random_guided_warp, i, batch[i], random_prototype_indices[i], full_dataset, slope_constraint, dtw_type, window) for i in range(N)]

        # Collect results
        for i, future in enumerate(futures):
            processed_data[i] = future.result()

    return processed_data

def random_guided_warp_shape(batch, full_dataset, batch_labels, full_dataset_labels, slope_constraint="symmetric", use_window=True):
    return random_guided_warp_multithreaded(
        batch=batch, 
        full_dataset=full_dataset, 
        batch_labels=batch_labels, 
        full_dataset_labels=full_dataset_labels, 
        slope_constraint=slope_constraint, 
        use_window=use_window, 
        dtw_type="shape"
    )

def window_slice(x, reduce_ratio=0.9):
    target_len = np.ceil(reduce_ratio * x.shape[1]).astype(int)
    if target_len >= x.shape[1]:
        return x

    starts = np.random.randint(low=0, high=x.shape[1] - target_len, size=(x.shape[0])).astype(int)
    ends = (target_len + starts).astype(int)

    new_indices = np.linspace(0, target_len, num=x.shape[1], dtype=int)
    ret = np.zeros_like(x)

    # Vectorized operation
    for dim in range(x.shape[2]):
        ret[:, :, dim] = np.array([np.interp(new_indices, np.arange(starts[i], ends[i]), x[i, starts[i]:ends[i], dim]) for i in range(x.shape[0])])

    return ret
    
#porposed approach discriminative_guided_warp_worker
def downsample(sequence, factor):
    # Downsample each channel independently
    return sequence[::factor, :]

def upsample(warped_sequence, num_channels=12, original_length=2934):
    if not isinstance(warped_sequence, np.ndarray):
        warped_sequence = np.array(warped_sequence)
    
    upsampled = np.zeros((original_length, num_channels), dtype=int)
    
    for i in range(num_channels):
        interpolated = np.interp(
            np.linspace(0, original_length - 1, num=original_length), 
            np.linspace(0, warped_sequence.shape[0] - 1, num=warped_sequence.shape[0]), 
            warped_sequence[:, i]
        )
        upsampled[:, i] = np.round(interpolated).astype(int)

    return upsampled

def discriminative_guided_warp_worker(i, pat, positive_prototypes, negative_prototypes, slope_constraint, dtw_type, window, orig_steps, x_shape):
    C = x_shape[2]  # Number of channels

    if len(positive_prototypes) == 0 or len(negative_prototypes) == 0:
        return pat, 0

    pos_k = len(positive_prototypes)
    neg_k = len(negative_prototypes)
    pos_aves = np.zeros(pos_k)
    neg_aves = np.zeros(pos_k)
    warp_amount = 0

    dtw_cache = {}  # Cache for storing DTW results

    # DTW calculations with caching
    for p, pos_prot in enumerate(positive_prototypes):
        for ps, pos_samp in enumerate(positive_prototypes):
            if p != ps:
                cache_key = tuple(sorted((p, ps)))
                if cache_key not in dtw_cache:
                    if dtw_type == "shape":
                        dtw_cache[cache_key] = shape_dtw(pos_prot, pos_samp, RETURN_VALUE, slope_constraint=slope_constraint, window=window)
                    else:
                        dtw_cache[cache_key] = dtw(pos_prot, pos_samp, RETURN_VALUE, slope_constraint=slope_constraint, window=window)
                
                pos_aves[p] += (1./(pos_k-1.)) * dtw_cache[cache_key]

        for ns, neg_samp in enumerate(negative_prototypes):
            cache_key = (p, ns)
            if cache_key not in dtw_cache:
                if dtw_type == "shape":
                    dtw_cache[cache_key] = shape_dtw(pos_prot, neg_samp, RETURN_VALUE, slope_constraint=slope_constraint, window=window)
                else:
                    dtw_cache[cache_key] = dtw(pos_prot, neg_samp, RETURN_VALUE, slope_constraint=slope_constraint, window=window)

            neg_aves[p] += (1./neg_k) * dtw_cache[cache_key]

    selected_id = np.argmax(neg_aves - pos_aves)
    selected_prototype = positive_prototypes[selected_id]
    if dtw_type == "shape":
        path = shape_dtw(selected_prototype, pat, RETURN_PATH, slope_constraint=slope_constraint, window=window)
    else:
        path = dtw(selected_prototype, pat, RETURN_PATH, slope_constraint=slope_constraint, window=window)

    # Time warp
    warped = pat[path[1]]

    warp_path_interp = np.interp(orig_steps, np.linspace(0, x_shape[1]-1., num=warped.shape[0]), path[1])
    warp_amount = np.sum(np.abs(orig_steps - warp_path_interp))
    warped_instance = np.zeros_like(pat)
    for dim in range(C):
        warped_instance[:, dim] = np.interp(orig_steps, np.linspace(0, x_shape[1]-1., num=warped.shape[0]), warped[:, dim]).T

    return warped_instance, warp_amount

def downsample(sequence, factor):
    return sequence[::factor, :]

def upsample_path(path, orig_length, downsampled_length):
    upsampled_path = []
    for dim_path in path:  # Assuming each dim_path is a list
        upsampled_path_dim = np.interp(
            np.linspace(0, orig_length - 1, num=orig_length),
            np.linspace(0, downsampled_length - 1, num=len(dim_path)),
            dim_path
        ).astype(int)
        upsampled_path.append(upsampled_path_dim)
    return upsampled_path

def discriminative_guided_warp_worker(i, pat, positive_prototypes, negative_prototypes, slope_constraint, dtw_type, window, orig_steps, x_shape):
    C = x_shape[2]  # Number of channels
    downsample_factor = 5
    downsampled_length = pat.shape[0] // downsample_factor

    if len(positive_prototypes) == 0 or len(negative_prototypes) == 0:
        return pat, 0

    # Downsample
    pat_downsampled = downsample(pat, downsample_factor)
    positive_downsampled = [downsample(p, downsample_factor) for p in positive_prototypes]
    negative_downsampled = [downsample(n, downsample_factor) for n in negative_prototypes]

    pos_k = len(positive_downsampled)
    neg_k = len(negative_downsampled)
    pos_aves = np.zeros(pos_k)
    neg_aves = np.zeros(pos_k)
    dtw_cache = {}  # Cache for storing DTW results

    # DTW calculations with caching
    for p, pos_prot in enumerate(positive_downsampled):
        for ps, pos_samp in enumerate(positive_downsampled):
            if p != ps:
                cache_key = tuple(sorted((p, ps)))
                if cache_key not in dtw_cache:
                    if dtw_type == "shape":
                        dtw_cache[cache_key] = shape_dtw(pos_prot, pos_samp, RETURN_VALUE, slope_constraint=slope_constraint, window=window)
                    else:
                        dtw_cache[cache_key] = dtw(pos_prot, pos_samp, RETURN_VALUE, slope_constraint=slope_constraint, window=window)
                
                pos_aves[p] += (1./(pos_k-1.)) * dtw_cache[cache_key]

        for ns, neg_samp in enumerate(negative_downsampled):
            cache_key = (p, ns)
            if cache_key not in dtw_cache:
                if dtw_type == "shape":
                    dtw_cache[cache_key] = shape_dtw(pos_prot, neg_samp, RETURN_VALUE, slope_constraint=slope_constraint, window=window)
                else:
                    dtw_cache[cache_key] = dtw(pos_prot, neg_samp, RETURN_VALUE, slope_constraint=slope_constraint, window=window)

            neg_aves[p] += (1./neg_k) * dtw_cache[cache_key]

    selected_id = np.argmax(neg_aves - pos_aves)
    selected_prototype_downsampled = positive_downsampled[selected_id]

    # Compute DTW path using the downsampled prototype
    if dtw_type == "shape":
        path_downsampled = shape_dtw(selected_prototype_downsampled, pat_downsampled, RETURN_PATH, slope_constraint=slope_constraint, window=window)
    else:
        path_downsampled = dtw(selected_prototype_downsampled, pat_downsampled, RETURN_PATH, slope_constraint=slope_constraint, window=window)

    # Upsample the DTW path
    path_upsampled = upsample_path(path_downsampled, pat.shape[0], downsampled_length)

    # Time warp using the upsampled path
    warped_instance = np.zeros_like(pat)
    for dim in range(C):
        # Validate dimension index and interpolate
        if dim < len(path_upsampled):
            warped_instance[:, dim] = np.interp(
                orig_steps, 
                np.linspace(0, pat.shape[0] - 1, num=len(path_upsampled[dim])),
                path_upsampled[dim]
            )

    # Calculate warp amount
    # Ensure path_upsampled has the correct dimension
    if len(path_upsampled) >= C:
        warp_amount = np.sum([np.abs(orig_steps - path_upsampled[dim]) for dim in range(C)])
    else:
        warp_amount = 0  # No valid warp path found

    return warped_instance, warp_amount

def discriminative_guided_warp(x, full_dataset, labels, full_dataset_labels, batch_size=6, slope_constraint="symmetric", use_window=True, dtw_type="normal", use_variable_slice=True, verbose=0, num_threads=34):
    if use_window:
        window = np.ceil(x.shape[1] / 10.).astype(int)
    else:
        window = None

    x = x.astype(np.float32)
    full_dataset = full_dataset.astype(np.float32)

    orig_steps = np.arange(x.shape[1])
    batch_labels = np.argmax(labels, axis=1) if labels.ndim > 1 else labels

    # Precompute the prototypes for each sample in x
    prototypes_info = []
    for i, label in enumerate(batch_labels):
        positive = np.where(full_dataset_labels == label)[0]
        negative = np.where(full_dataset_labels != label)[0]
        positive_prototypes = full_dataset[np.random.choice(positive, min(len(positive), batch_size//2), replace=False)] if len(positive) > 0 else np.array([])
        negative_prototypes = full_dataset[np.random.choice(negative, min(len(negative), batch_size//2), replace=False)] if len(negative) > 0 else np.array([])
        prototypes_info.append((positive_prototypes, negative_prototypes))

    # Multithreading
    ret = np.zeros_like(x)
    warp_amount = np.zeros(x.shape[0])

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(discriminative_guided_warp_worker, i, x[i], *prototypes_info[i], slope_constraint, dtw_type, window, orig_steps, x.shape) for i in range(x.shape[0])]

        for i, future in enumerate(futures):
            ret[i], warp_amount[i] = future.result()

    # Post-processing
    if use_variable_slice:
        max_warp = np.max(warp_amount)
        if max_warp == 0:
            ret = window_slice(ret, reduce_ratio=0.9)
        else:
            for i, pat in enumerate(ret):
                reduce_ratio = 0.9 + 0.1 * warp_amount[i] / max_warp
                ret[i] = window_slice(pat[np.newaxis, :, :], reduce_ratio=reduce_ratio)[0]

    return ret

def discriminative_guided_warp_shape(x, labels, full_dataset, full_dataset_labels, batch_size=6, slope_constraint="symmetric", use_window=True, use_variable_slice=True, verbose=0, num_threads=18):
    return discriminative_guided_warp(x, labels, full_dataset, full_dataset_labels, batch_size, slope_constraint, use_window, dtw_type="shape", use_variable_slice=use_variable_slice, verbose=verbose, num_threads=num_threads)

##############
# Dataloader #
##############

#this function is used as there are some data augmentation strategies that
#require other labeled examples to extrapolate, to limit memory usage
#and permit mutlithreading keep only examples with over N examples
#also remove singlets as they are just the same sample and thus
#taking less memory
def select_indices(X, Y, target_count=25):
    # Find unique rows and their counts
    unique_rows, counts = np.unique(Y, axis=0, return_counts=True)
    # Mask for rows that appear more than once
    mask = counts > 1

    # Select only those rows and counts
    unique_rows = unique_rows[mask]
    counts = counts[mask]

    # Get indices where Y matches any of the unique rows
    selected_indices = np.array([], dtype=int)
    for row, count in tzip(unique_rows, counts):
        row_indices = np.where((Y == row).all(axis=1))[0]
        if count < target_count:
            selected_indices = np.append(selected_indices, row_indices)
        else:
            selected_indices = np.append(selected_indices, np.random.choice(row_indices, size=target_count, replace=False))

    # Use the selected indices to filter X and Y
    X_selected = X[selected_indices]
    Y_selected = Y[selected_indices]

    # Print the percentage and number of rows remaining
    percentage_remaining = (len(selected_indices) / len(Y)) * 100
    print(f"Percentage of rows remaining: {percentage_remaining:.2f}%")
    print(f"Number of rows remaining: {len(selected_indices)}")

    return X_selected, Y_selected

def val_collate_fn(batch):
    # Unzip the batch into data and labels
    data = [x for x, y in batch]
    labels = [y for x, y in batch]

    # Convert to numpy arrays
    data = np.squeeze(np.stack(data, axis=0))
    labels = np.stack(labels, axis=0)

    # Find rows where any element is not between 0 and 1
    invalid_rows = np.any((labels < 0) | (labels > 1), axis=1)

    # Get indices of invalid rows
    invalid_indices = np.where(invalid_rows)[0]

    # Filter out invalid rows from both X and Y matrices
    data = np.delete(data, invalid_indices, axis=0)  
    labels = np.delete(labels, invalid_indices, axis=0) 
    return torch.tensor(data, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)

def custom_collate_fn(batch, transform, transform_prob):
    #print('collate')
    #print(len(batch))
    #print(len(batch[0]))

    #print(batch[1][0].shape)
    #print(batch[1][1].shape)

    # Unzip the batch into data and labels
    data = [x for x, y in batch]
    labels = [y for x, y in batch]

    # Convert to numpy arrays
    data = np.squeeze(np.stack(data, axis=0))
    labels = np.stack(labels, axis=0)

    # Find rows where any element is not between 0 and 1
    invalid_rows = np.any((labels < 0) | (labels > 1), axis=1)

    # Get indices of invalid rows
    invalid_indices = np.where(invalid_rows)[0]

    # Filter out invalid rows from both X and Y matrices
    data = np.delete(data, invalid_indices, axis=0)  
    labels = np.delete(labels, invalid_indices, axis=0) 
    

    #print(data.shape)
    #print(labels.shape)

    # Determine the number of samples to transform
    batch_size = len(data)
    num_to_transform = int(transform_prob * batch_size)
    indices_to_transform = np.random.choice(batch_size, num_to_transform, replace=False)


    transform_ = {'window_warp_multithreaded':window_warp_multithreaded,
                  'window_slice_multithreaded':window_slice_multithreaded,
                  'time_warp_multithreaded': time_warp_multithreaded,
                  'magnitude_warp_uniform_multithreaded':magnitude_warp_uniform_multithreaded,
                  'beat_permutation':beat_permutation,
                  'scaling':scaling,
                  'jitter':jitter,
                  }
    
    transform = transform_[transform]

    #print(type(data))
    #print(data.shape)

    # Apply the transformation to the selected samples 
    if transform is not None:
        #data = np.swapaxes(data, -2,-1)
        data[indices_to_transform] = transform(data[indices_to_transform])
        data = np.swapaxes(data, -2,-1)

    return torch.tensor(np.swapaxes(data, -2,-1), dtype=torch.float32), torch.tensor(labels, dtype=torch.long)


def custom_collate_fn_label(batch, ds, ds_labels, transform, transform_prob):

    print(batch.shape)
    # Unzip the batch into data and labels
    data, labels = zip(*batch)

    print(data.shape)
    # Convert to numpy arrays
    data = np.array(data)
    labels = np.array(labels)

    # Determine the number of samples to transform
    batch_size = len(data)
    num_to_transform = int(transform_prob * batch_size)
    indices_to_transform = np.random.choice(batch_size, num_to_transform, replace=False)


    # Apply the transformation to the selected samples 
    if transform is not None:
        data[indices_to_transform] = transform(data[indices_to_transform],ds,labels[indices_to_transform],ds_labels)

    return torch.tensor(np.swapaxes(data, -2,-1), dtype=torch.float32), torch.tensor(labels, dtype=torch.long)

class MyCustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        return x, y
    


#dataset = MyCustomDataset(data, labels)
#data_loader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=lambda batch: custom_collate_fn(batch, my_transform, 0.5))


class ECGDataset(Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform
        
    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]       
        return np.swapaxes(x.astype(np.float32), -1,-2), y.astype(np.float32)
    
    def __len__(self):
        return len(self.data)
    

def get_shape_dtype_from_npy_file(npy_file_path):
    with open(npy_file_path, 'rb') as f:
        # Read the magic string to get version number
        version = read_magic(f)

        # Read the array header which contains the shape and dtype
        shape, fortran_order, dtype = _read_array_header(f, version)
        
    return shape, dtype

class CustomDataset(Dataset):
    def __init__(self, features_memmap_path, labels_memmap_path, features_shape, labels_shape, pos_to_drop, dtype=np.float16):
        """
        Initialize the dataset.
        :param features_memmap_path: Path to the .npy file for features.
        :param labels_memmap_path: Path to the .npy file for labels.
        :param features_shape: Shape of the features dataset array.
        :param labels_shape: Shape of the labels dataset array.
        :param pos_to_drop: Positions (columns) to drop from the Y labels.
        :param dtype: Data type of the datasets.
        """
        self.features_memmap_path = features_memmap_path
        self.labels_memmap_path = labels_memmap_path
        self.features_shape = features_shape
        self.labels_shape = labels_shape
        self.pos_to_drop = pos_to_drop
        self.dtype = dtype

    def __len__(self):
        return self.features_shape[0]  # Assuming first dimension is number of samples

    def __getitem__(self, idx):
        # Memory-map the file in read-only mode for features and labels
        #features = np.memmap(self.features_memmap_path, dtype=self.dtype, mode='r', shape=self.features_shape)[idx]
        features = np.load(self.features_memmap_path, mmap_mode='r')[idx]
        labels = np.load(self.labels_memmap_path, mmap_mode='r')[idx]


        # Drop specified columns from labels
        labels = np.delete(labels, self.pos_to_drop, axis=0)  # axis=0 for dropping columns

        # Convert to torch tensors
        features_tensor = torch.from_numpy(np.expand_dims(features, axis=0)).float()
        labels_tensor = torch.from_numpy(labels).float()


        return features_tensor, labels_tensor


#########
# other #
#########

def set_seed(seed):
    """
    Set the seed for random number generators in PyTorch and NumPy, and CUDA if available.

    Parameters:
    - seed (int): Seed value for reproducibility.
    """
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    # If using cudnn backend
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    torch.cuda.manual_seed_all(seed) 

    np.random.seed(seed)


def filtering(data, name, lowcut, highcut,fs=250, order=2):
    N = data.shape[0] # Replace with your actual N


    if lowcut == -1:
        b, a = butter(order, highcut, btype='lowpass', fs=fs)

    if highcut == -1:
        b, a = butter(order, lowcut, btype='highpass', fs=fs)

    else:
        b, a = butter(order, [lowcut,highcut], btype='bandpass', fs=fs)


    def apply_filter(data_slice):
        """Applies the bandpass filter to a slice of the data."""
        filtered_slice = np.empty_like(data_slice).astype(np.float16)
        for i in range(data_slice.shape[0]):
            for j in range(data_slice.shape[-1]):
                filtered_slice[i, :, j] = filtfilt(b, a, data_slice[i, :, j])
        return filtered_slice

    # Divide data into chunks for parallel processing
    num_processes = cpu_count()
    chunk_size = N // num_processes
    chunks = [data[i:i + chunk_size].astype(np.float16) for i in range(0, N, chunk_size)]

    # Perform parallel processing with progress tracking
    with Pool(num_processes) as pool:
        results = list(tqdm(pool.imap(apply_filter, chunks), total=len(chunks), desc=f'Filtering at highcut:{lowcut} lowcut:{highcut} for {name}'))

    # Reassemble the results
    return np.concatenate(results, axis=0)


import copy
import torch
import copy
import torch

import copy
import torch

class EMAWeightUpdater:
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow_params = copy.deepcopy(model.state_dict())

    def update(self):
        model_params = self.model.state_dict()
        
        # Update shadow parameters with EMA formula
        with torch.no_grad():
            for name, param in model_params.items():
                if param.dtype.is_floating_point:
                    if name not in self.shadow_params:
                        self.shadow_params[name] = param.clone()
                    else:
                        shadow_param = self.shadow_params[name]
                        new_value = param * (1.0 - self.decay) + shadow_param * self.decay
                        self.shadow_params[name].copy_(new_value)

    def apply_shadow(self):
        original_state_dict = self.model.state_dict()
        self.model.load_state_dict(self.shadow_params, strict=True)
        return original_state_dict

    def restore_original(self, original_state_dict):
        self.model.load_state_dict(original_state_dict)

def logits_to_hard_labels(logits, threshold=0.5):
    """
    Convert logits from a multi-label model to hard labels.

    :param logits: A tensor of logits from the model.
    :param threshold: Threshold to convert probabilities to binary values. Defaults to 0.5.
    :return: A tensor of hard labels.
    """
    # Apply sigmoid to convert logits to probabilities
    probabilities = torch.sigmoid(logits)

    # Apply threshold to convert probabilities to binary values
    hard_labels = (probabilities > threshold).int()  # Using .int() to convert boolean to integers (0, 1)
    return hard_labels
