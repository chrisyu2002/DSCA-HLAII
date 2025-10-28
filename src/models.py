#!/usr/bin/env python3
# -*- coding: utf-8
"""
Created on 2025/09/18
@author yhj

"""

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Optional, Mapping, Tuple
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.evaluation import acc_and_auc, get_perfomance
from src.early_stopping import EarlyStopping
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
__all__ = ['Model']

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

class Model(object):
    """

    """
    def __init__(self, network, model_path, **kwargs):
        self.model = self.network = network(**kwargs).to(device)
        self.loss_fn, self.model_path = nn.BCELoss(), Path(model_path)
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        self.optimizer = None
        self.training_state = {}
        self.early_stopping = None
        self.scheduler = None

    def get_scores(self, inputs, **kwargs):
        scores = self.model(*(x.to(device) for x in inputs), **kwargs)
        scores = scores.squeeze(1)
        return scores

    def loss_and_backward(self, scores, targets):
        loss = self.loss_fn(scores, targets.to(device))
        loss.backward()
        return loss

    def train_step(self, inputs: Tuple[torch.Tensor, torch.Tensor], targets: torch.Tensor, **kwargs):
        self.optimizer.zero_grad()
        self.model.train()
        loss = self.loss_and_backward(self.get_scores(inputs, **kwargs), targets)
        self.optimizer.step(closure=None)
        return loss.item()

    @torch.no_grad()
    def predict_step(self, inputs: Tuple[torch.Tensor, torch.Tensor], **kwargs):
        self.model.eval()
        return self.get_scores(inputs, **kwargs).cpu()

    def get_optimizer(self, optimizer_cls='Adadelta', weight_decay=1e-3, **kwargs):
        if isinstance(optimizer_cls, str):
            optimizer_cls = getattr(torch.optim, optimizer_cls)
        self.optimizer = optimizer_cls(self.model.parameters(), weight_decay=weight_decay, **kwargs)

    def train(self, train_loader: DataLoader, valid_loader: DataLoader, opt_params: Optional[Mapping] = (),
              num_epochs=20, verbose=True, **kwargs):
        self.get_optimizer(**dict(opt_params))
        # self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=3, verbose=True)
        self.training_state['best'] = 0.0
        self.early_stopping = EarlyStopping(model_path = self.model_path)
        for epoch_idx in range(num_epochs):
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch_idx}, Learning Rate: {current_lr}")
            train_loss = 0.0
            for inputs, targets in tqdm(train_loader, desc=f'Epoch {epoch_idx}', leave=False, dynamic_ncols=True):
                train_loss += self.train_step(inputs, targets, **kwargs) * len(targets)
            train_loss /= len(train_loader.dataset)
            
            if(self.valid(valid_loader, verbose, epoch_idx, train_loss)):
                break
        
    def valid(self, valid_loader, verbose, epoch_idx, train_loss, **kwargs):
        model = self.model
        
        num_correct = 0
        y_true_val_list = []
        y_pred_val_list = []
        ouputs_int_list = []

        valid_loss = 0.0
        total_samples = 0
        self.model.eval()

        with torch.no_grad():
            for inputs, targets in tqdm(valid_loader, desc=f'Valid Epoch {epoch_idx}', leave=False, dynamic_ncols=True):
                scores = self.get_scores(inputs, **kwargs)
                
                loss = self.loss_fn(scores, targets.to(device))
                valid_loss += loss.item() * len(targets)
                total_samples += len(targets)

                y_pred_val, y_true_val, num_correct_tmp, ouputs_int = acc_and_auc(scores.cpu().numpy(), targets.numpy())
                y_pred_val_list += y_pred_val
                y_true_val_list += y_true_val
                ouputs_int_list += ouputs_int
                num_correct += num_correct_tmp

        valid_loss /= total_samples
        # self.scheduler.step(valid_loss)
        val_auc = roc_auc_score(np.array(y_true_val_list), np.array(y_pred_val_list))
        _, _, MCC, ACC, F1, precision, recall, _ = get_perfomance(y_true_val_list, ouputs_int_list)
        pr_precision, pr_recall, _ = precision_recall_curve(y_true_val_list, y_pred_val_list)
        AUPR = auc(pr_recall, pr_precision)
        PCC = np.corrcoef(y_true_val_list, y_pred_val_list)[0, 1]
        
        print(  f'Epoch: {epoch_idx} '
                f'train loss: {train_loss:.5f} '
                f'valid loss: {valid_loss:.5f} ')
        print('Precision: ' , '%.3f'%precision, '\tRecall: ', '%.3f'%recall, 
            '\tMCC: ', '%.3f'%MCC, '\tACC: ' , '%.3f'%ACC, 
            '\tF1: ', '%.3f'%F1, '\nAUC: ', '%.3f'%val_auc,
            '\tAUPR: ', '%.3f'%AUPR,
            '\tPCC: ', '%.3f'%PCC)

        self.early_stopping(valid_loss, model)
        if self.early_stopping.early_stop:
            print("Early stopping")
            return True
        return False

    def predict(self, data_loader: DataLoader, valid=False, **kwargs):
        if not valid:
            self.load_model()
        return np.concatenate([self.predict_step(data_x, **kwargs)
                               for data_x, _ in tqdm(data_loader, leave=False, dynamic_ncols=True)], axis=0)
        
    def predict_core(self, pep_one_hot, pep_esm, hla_one_hot, hla_esm, **kwargs):
        pep_one_hot, pep_esm, hla_one_hot, hla_esm = torch.from_numpy(pep_one_hot), torch.from_numpy(pep_esm), torch.from_numpy(hla_one_hot), torch.from_numpy(hla_esm)
        data_x = [pep_one_hot, pep_esm, hla_one_hot, hla_esm]
        self.load_model()
        return np.concatenate([self.predict_step(data_x, **kwargs)], axis=0)
    
    def save_model(self):
        torch.save(self.model.state_dict(), self.model_path)

    def load_model(self):
        self.model.load_state_dict(torch.load(self.model_path))