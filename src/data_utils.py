#!/usr/bin/env python3
# -*- coding: utf-8
"""
Created on 2025/09/18
@author yhj

"""

__all__ = ['ACIDS', 'get_hla_name_seq', 'get_data', 'get_binding_data', 'get_seq2logo_data']
ACIDS = '0-ACDEFGHIKLMNPQRSTVWY'
from src.generate_core_feature import *

def get_hla_name_seq(hla_name_seq_file):
    hla_dict = {}
    with open(hla_name_seq_file) as fp:
        for line in fp:
            parts = line.strip().split('\t', 1)
            if len(parts) == 2:
                hla_name, hla_seq = parts
                hla_dict[hla_name] = hla_seq
    
    class HLA_Sequence_Getter:
        def __getitem__(self, name):
            if '-' in name:
                parts = name.split('-')
                return ''.join(hla_dict[part] for part in parts)
            else:
                return hla_dict[name]
    
    return HLA_Sequence_Getter()


import numpy as np

def load_allele_mapping(txt_file, npy_file):
    allele_dict = {}
    with open(txt_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for idx, line in enumerate(lines):
            parts = line.strip().split()
            if len(parts) != 2:
                raise ValueError(f"Invalid line format at line {idx + 1}: {line}")
            allele_name, _ = parts
            allele_dict[allele_name] = idx

    npy_data = np.load(npy_file)
    if npy_data.shape[0] != len(allele_dict):
        raise ValueError("Mismatch between number of alleles in TXT and NPY dimensions.")

    def get_allele_array(allele_name):
        if allele_name not in allele_dict:
            raise KeyError(f"Allele {allele_name} not found in TXT file.")
        return npy_data[allele_dict[allele_name]]

    return get_allele_array

allele_lookup = load_allele_mapping('data/hla_dict/hla_full_seq_dict.txt', 'data/hla_dict/hla_esm_dict.npy')


def get_data_core(hla_name_seq, data_cnf):

    data_file = data_cnf['core']
    pep_seqs = []
    hla_seqs = []
    cores = []
    with open(data_file) as fp:
        for i, line in enumerate(fp):
            id, hla_alpha_name, hla_beta_name, pesudo_seq, peptide_seq, core = line.split()
            hla_name = hla_alpha_name + '-' + hla_beta_name
            hla_seq = hla_name_seq[hla_name]
            pep_seqs.append(peptide_seq)
            hla_seqs.append(hla_seq)
            
    pep_one_hot_list = generate_pep_one_hot(pep_seqs)
    hla_one_hot_list = generate_hla_one_hot(hla_seqs)
    
    pep_esm_list = generate_pep_esm(pep_seqs)
    hla_esm_list = generate_hla_esm_alpha_beta(hla_seqs)

    return pep_seqs, hla_seqs, pep_one_hot_list, hla_one_hot_list, pep_esm_list, hla_esm_list, cores

def get_data_eval(hla_name_seq, data_cnf, mode):
    data_list = []
    data_file = data_cnf[mode]

    pep_esm_file = data_cnf[f'pep_esm_{mode}']
    hla_esm_file = data_cnf[f'hla_esm_{mode}']
    
    pep_esm_array = np.load(pep_esm_file, mmap_mode='r')
    hla_esm_array = np.load(hla_esm_file, mmap_mode='r')


    with open(data_file) as fp:
        for i, line in enumerate(fp):
            peptide_seq, hla_alpha_name, hla_beta_name, score = line.split()
            hla_name = hla_alpha_name + '-' + hla_beta_name
            pep_esm_data = pep_esm_array[i]
            hla_esm_data = hla_esm_array[i]

            # 添加到data_list（避免存储全部数据）
            data_list.append((
                hla_name,
                peptide_seq,
                hla_name_seq[hla_name],
                float(score),
                pep_esm_data,
                hla_esm_data
            ))
    
    return data_list


def get_data_train(hla_name_seq, data_cnf):
    data_list = []
    data_file = data_cnf['train']
    pep_esm_files = [data_cnf['pep_esm_train_1'], data_cnf['pep_esm_train_2'], data_cnf['pep_esm_train_3'], data_cnf['pep_esm_train_4'], data_cnf['pep_esm_train_5']]
    hla_esm_files = [data_cnf['hla_esm_train_1'], data_cnf['hla_esm_train_2'], data_cnf['hla_esm_train_3'], data_cnf['hla_esm_train_4'], data_cnf['hla_esm_train_5']]
    
    pep_esm_arrays = [np.load(f, mmap_mode='r') for f in pep_esm_files]
    hla_esm_arrays = [np.load(f, mmap_mode='r') for f in hla_esm_files]
    
    pep_esm_shapes = [arr.shape[0] for arr in pep_esm_arrays]
    hla_esm_shapes = [arr.shape[0] for arr in hla_esm_arrays]
    
    pep_esm_offsets = np.cumsum([0] + pep_esm_shapes[:-1])
    hla_esm_offsets = np.cumsum([0] + hla_esm_shapes[:-1])
    
    
    with open(data_file) as fp:
        for i, line in enumerate(fp):
            peptide_seq, hla_alpha_name, hla_beta_name, score = line.split()
            hla_name = hla_alpha_name + '-' + hla_beta_name
            
            pep_file_idx = np.searchsorted(pep_esm_offsets, i, side='right') - 1
            pep_local_idx = i - pep_esm_offsets[pep_file_idx]
            pep_esm_data = pep_esm_arrays[pep_file_idx][pep_local_idx]
            
            hla_file_idx = np.searchsorted(hla_esm_offsets, i, side='right') - 1
            hla_local_idx = i - hla_esm_offsets[hla_file_idx]
            hla_esm_data = hla_esm_arrays[hla_file_idx][hla_local_idx]
            
            data_list.append((
                hla_name,
                peptide_seq,
                hla_name_seq[hla_name],
                float(score),
                pep_esm_data,
                hla_esm_data
            ))
    
    return data_list
