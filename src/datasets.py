#!/usr/bin/env python3
# -*- coding: utf-8
"""
Created on 2025/09/18
@author yhj

"""

import numpy as np
from torch.utils.data.dataset import Dataset
from src.data_utils import ACIDS

__all__ = ['HLAIIDataset']

class HLAIIDataset(Dataset):
    """

    """
    def __init__(self, data_list, peptide_len=32, hla_len=34):
        self.data_list = data_list
        self.peptide_len = peptide_len
        self.hla_len = hla_len

    def __getitem__(self, item):
        hla_name, peptide_seq, hla_seq, score, pep_esm, hla_esm = self.data_list[item]
        if len(peptide_seq) > self.peptide_len:
            peptide_seq = peptide_seq[:self.peptide_len]
        elif len(peptide_seq) < self.peptide_len:
            peptide_seq = peptide_seq.ljust(self.peptide_len, 'X')

        peptide_x = [ACIDS.index(x if x in ACIDS else '-') for x in peptide_seq]
        hla_x = [ACIDS.index(x if x in ACIDS else '-') for x in hla_seq]
        peptide_x = np.array(peptide_x)
        hla_x = np.array(hla_x)
        
        peptide_one_hot = np.zeros((len(peptide_x), len(ACIDS)), dtype=np.float32)
        peptide_one_hot[np.arange(len(peptide_x)), peptide_x] = 1
        
        hla_one_hot = np.zeros((len(hla_x), len(ACIDS)), dtype=np.float32)
        hla_one_hot[np.arange(len(hla_x)), hla_x] = 1
        
        return (peptide_one_hot, pep_esm, hla_one_hot, hla_esm), np.float32(score)

    def __len__(self):
        return len(self.data_list)