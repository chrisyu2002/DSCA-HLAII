#!/usr/bin/env python3
# -*- coding: utf-8
"""
Created on 2025/09/18
@author yhj

"""

import torch
import numpy as np
from tqdm import tqdm
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig
import torch.nn.functional as F
    
ACIDS = '0-ACDEFGHIKLMNPQRSTVWY'

def generate_pep_esm(sequences, client):

    all_embeddings = []

    batch_size = 50

    total_batches = len(sequences) // batch_size + (1 if len(sequences) % batch_size != 0 else 0)

    for i in tqdm(range(0, len(sequences), batch_size), desc="Processing Batches", total=total_batches):
        batch_sequences = sequences[i:i + batch_size]
        batch_embeddings = []
        
        for sequence in batch_sequences:
            protein = ESMProtein(sequence)
            
            protein_tensor = client.encode(protein)
            logits_output = client.logits(
                protein_tensor, LogitsConfig(sequence=True, return_embeddings=True)
            )

            trimmed_embeddings = logits_output.embeddings[:, 1:-1, :]
            current_length = trimmed_embeddings.shape[1]

            trimmed_embeddings = trimmed_embeddings.squeeze(0)
            
            current_len = trimmed_embeddings.size(0)
            if current_len >= 32:
                padded_embeddings = trimmed_embeddings[:32, :]
            elif current_len < 32:
                pad_rows = 32 - trimmed_embeddings.size(0)
                padded_embeddings = F.pad(trimmed_embeddings, (0, 0, 0, pad_rows), mode='constant', value=0)
            
            batch_embeddings.append(padded_embeddings.to(torch.float32).cpu().numpy())
        
        all_embeddings.extend(batch_embeddings)

    all_embeddings_array = np.array(all_embeddings)
    return all_embeddings_array

def generate_hla_esm_alpha_beta(sequences, client):
    """
    sequences: list of strings, each string length = 200
    return: numpy array of shape [batch_size, 2, 100, 1152]
    """
    all_embeddings = []

    batch_size = 50

    total_batches = len(sequences) // batch_size + (1 if len(sequences) % batch_size != 0 else 0)

    for i in tqdm(range(0, len(sequences), batch_size), desc="Processing Batches", total=total_batches):
        batch_sequences = sequences[i:i + batch_size]
        
        batch_embeddings = []

        for sequence in batch_sequences:

            seq_part1 = sequence[:100]
            seq_part2 = sequence[100:]

            embeddings_parts = []

            for part in [seq_part1, seq_part2]:
                protein = ESMProtein(part)
                protein_tensor = client.encode(protein)

                logits_output = client.logits(
                    protein_tensor, LogitsConfig(sequence=True, return_embeddings=True)
                )

                # 去除CLS和EOS
                trimmed_embeddings = logits_output.embeddings[:, 1:-1, :]  # shape [1, seq_len, 1152]
                trimmed_embeddings = trimmed_embeddings.squeeze(0)  # shape [seq_len, 1152]

                # 理论上seq_len应该就是100
                current_len = trimmed_embeddings.size(0)
                if current_len >= 100:
                    padded_embeddings = trimmed_embeddings[:100, :]
                else:
                    pad_rows = 100 - current_len
                    padded_embeddings = F.pad(trimmed_embeddings, (0, 0, 0, pad_rows), mode='constant', value=0)

                embeddings_parts.append(padded_embeddings.to(torch.float32).cpu().numpy())  # shape [100, 1152]

            combined = np.stack(embeddings_parts, axis=0)
            batch_embeddings.append(combined)

        # shape [batch_size, 2, 100, 1152]
        all_embeddings.extend(batch_embeddings)

    all_embeddings_array = np.array(all_embeddings)  # shape [total_size, 2, 100, 1152]
    return all_embeddings_array

def generate_hla_esm(sequences, client):

    all_embeddings = []

    batch_size = 50

    total_batches = len(sequences) // batch_size + (1 if len(sequences) % batch_size != 0 else 0)

    for i in tqdm(range(0, len(sequences), batch_size), desc="Processing Batches", total=total_batches):
        batch_sequences = sequences[i:i + batch_size]
        batch_embeddings = []
        
        for sequence in batch_sequences:
            protein = ESMProtein(sequence)
            
            protein_tensor = client.encode(protein)
            logits_output = client.logits(
                protein_tensor, LogitsConfig(sequence=True, return_embeddings=True)
            )

            trimmed_embeddings = logits_output.embeddings[:, 1:-1, :]
            current_length = trimmed_embeddings.shape[1]

            trimmed_embeddings = trimmed_embeddings.squeeze(0)
            
            current_len = trimmed_embeddings.size(0)
            if current_len >= 200:
                padded_embeddings = trimmed_embeddings[:200, :]
            elif current_len < 200:
                pad_rows = 200 - trimmed_embeddings.size(0)
                padded_embeddings = F.pad(trimmed_embeddings, (0, 0, 0, pad_rows), mode='constant', value=0)
            
            batch_embeddings.append(padded_embeddings.to(torch.float32).cpu().numpy())
        
        all_embeddings.extend(batch_embeddings)

    all_embeddings_array = np.array(all_embeddings)
    return all_embeddings_array


def generate_pep_one_hot(sequences):
    
    all_peptide_one_hot = []

    for peptide_seq in sequences:
        if len(peptide_seq) > 32:
            peptide_seq = peptide_seq[:32]
        elif len(peptide_seq) < 32:
            peptide_seq = peptide_seq.ljust(32, 'X')

        peptide_x = [ACIDS.index(x if x in ACIDS else '-') for x in peptide_seq]
        peptide_x = np.array(peptide_x)
        
        peptide_one_hot = np.zeros((len(peptide_x), len(ACIDS)), dtype=np.float32)
        peptide_one_hot[np.arange(len(peptide_x)), peptide_x] = 1
        
        all_peptide_one_hot.append(peptide_one_hot)

    all_encoded_sequences = np.stack(all_peptide_one_hot)
    return all_encoded_sequences


def generate_hla_one_hot(sequences):

    all_hla_one_hot = []

    for hla_seq in sequences:

        hla_x = [ACIDS.index(x if x in ACIDS else '-') for x in hla_seq]
        hla_x = np.array(hla_x)
        
        hla_one_hot = np.zeros((len(hla_x), len(ACIDS)), dtype=np.float32)
        hla_one_hot[np.arange(len(hla_x)), hla_x] = 1
        
        all_hla_one_hot.append(hla_one_hot)

    all_encoded_sequences = np.stack(all_hla_one_hot)
    return all_encoded_sequences