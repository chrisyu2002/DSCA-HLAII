#!/usr/bin/env python3
# -*- coding: utf-8
"""
Created on 2025/09/18
@author yhj

"""

import click
import numpy as np
import math
import csv
from functools import partial
from pathlib import Path
from ruamel.yaml import YAML
from sklearn.model_selection import train_test_split
from torch.utils.data.dataloader import DataLoader
from logzero import logger
from sklearn.model_selection import KFold

from src.datasets import HLAIIDataset
from src.models import Model
from src.networks import DSCA_HLAII
from src.data_utils import get_hla_name_seq
from src.evaluation import output_res
from src.data_utils import get_data_train
from src.data_utils import get_data_eval
from src.data_utils import get_data_core
from src.generate_core_feature import *

def train(model, data_cnf, model_cnf, train_data, valid_data=None, random_state=1240):
    logger.info(f'Start training model {model.model_path}')
    if valid_data is None:
        train_data, valid_data = train_test_split(train_data, test_size=data_cnf.get('valid', 1000),
                                                  random_state=random_state)
    train_loader = DataLoader(HLAIIDataset(train_data), batch_size=model_cnf['train']['batch_size'], shuffle=True, num_workers=4)
    valid_loader = DataLoader(HLAIIDataset(valid_data), batch_size=model_cnf['valid']['batch_size'], num_workers=4)
    model.train(train_loader, valid_loader, **model_cnf['train'])
    logger.info(f'Finish training model {model.model_path}')

def test(model, model_cnf, test_data, predict_res=False):
    if predict_res is False:
        data_loader = DataLoader(HLAIIDataset(test_data), batch_size=model_cnf['test']['batch_size'])
        return model.predict(data_loader)
    else:
        data_loader = DataLoader(HLAIIDataset(test_data), batch_size=model_cnf['test']['batch_size'])
        return model.predict_res(data_loader)
        
def get_core(model, model_cnf, test_data):
    CUTOFF = 1.0 - math.log(500, 50000)
    data_loader = DataLoader(HLAIIDataset(test_data), batch_size=model_cnf['test']['batch_size'])
    scores = model.predict(data_loader,core=False)
    binding_core_scores = model.predict(data_loader,core=True)
    binding_core_pos = binding_core_scores.argmax(axis=-1)
    indices = np.where(scores > CUTOFF)[0]
    cores = binding_core_pos[indices]
    return indices, cores

@click.command()
@click.option('-d', '--data-cnf', type=click.Path(exists=True), default='configure/data.yaml')
@click.option('-m', '--model-cnf', type=click.Path(exists=True), default='configure/dsca-hlaii.yaml')
@click.option('--mode', type=click.Choice(('input', 'train', 'cv_train', 'cv_eval', 'binding_core')), default='input')
@click.option('-n', '--num_models', default=5)

def main(data_cnf, model_cnf, mode, num_models):
    yaml = YAML(typ='safe')
    data_cnf, model_cnf = yaml.load(Path(data_cnf)), yaml.load(Path(model_cnf))
    model_name = model_cnf['name']
    logger.info(f'Model Name: {model_name}')
    model_path = Path(model_cnf['path'])/f'{model_name}.pt'
    result_path = Path(data_cnf['results'])/f'{model_name}.txt'

    hla_name_seq = get_hla_name_seq(data_cnf['hla_full_seq'])
    get_data_fn_train = partial(get_data_train, hla_name_seq = hla_name_seq)
    get_data_fn_eval = partial(get_data_eval, hla_name_seq = hla_name_seq)
    get_data_fn_core = partial(get_data_core, hla_name_seq = hla_name_seq)

    if mode == 'input':
        input_path = Path(data_cnf['input'])/f'input-test.txt'
        result_path =  Path(data_cnf['results'])/f'input-test-{model_name}.txt'
        CUTOFF = 1.0 - math.log(500, 50000)

        pep_seq, hla_alpha_name, hla_beta_name, hla_alpha_seq, hla_beta_seq = [], [], [], [], []
        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                pep_seq.append(parts[0])
                hla_alpha_name.append(parts[1])
                hla_beta_name.append(parts[2])
                hla_alpha_seq.append(hla_name_seq[parts[1]])
                hla_beta_seq.append(hla_name_seq[parts[2]])

        hla_seq = [a + b for a, b in zip(hla_alpha_seq, hla_beta_seq)]
        pep_one_hot_list = generate_pep_one_hot(pep_seq)
        hla_one_hot_list = generate_hla_one_hot(hla_seq)
        client = ESMC.from_pretrained("esmc_600m").to("cpu").eval()
        pep_esm_list = generate_pep_esm(pep_seq, client)
        hla_esm_list = generate_hla_esm_alpha_beta(hla_seq, client)

        model = Model(DSCA_HLAII, model_path=model_path.with_stem(f'{model_path.stem}-train0'))
        scores = model.predict_core(
            pep_one_hot_list, pep_esm_list, hla_one_hot_list, hla_esm_list, core=False
        )
        core_scores = model.predict_core(
            pep_one_hot_list, pep_esm_list, hla_one_hot_list, hla_esm_list, core=True
        )
        core_positions = core_scores.argmax(axis=-1)
        
        with open(result_path, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["PeptideSeq", "HLA_Alpha", "HLA_Beta",
                            "Score", "Presentation", "Binding_core"])
            
            # 遍历整个列表
            for i in range(len(pep_seq)):
                peptide = pep_seq[i]
                hla_alpha = hla_alpha_name[i]
                hla_beta = hla_beta_name[i]
                score = scores[i]
                presentation = "true" if score >= CUTOFF else "false"

                if presentation == "true":
                    core_pos = core_positions[i]
                    if core_pos + 9 <= len(peptide):
                        binding_core = peptide[core_pos:core_pos + 9]
                    else:
                        binding_core = peptide[core_pos:]
                else:
                    binding_core = "NaN!"

                writer.writerow([
                    str(peptide),
                    str(hla_alpha),
                    str(hla_beta),
                    "%.3f" % score,
                    presentation,
                    str(binding_core)
                ])
            
    elif mode == 'train':
        
        train_data = get_data_fn_train(data_cnf=data_cnf)
        valid_data = None

        for model_id in range(0, num_models):
            model = Model(DSCA_HLAII, model_path=model_path.with_stem(f'{model_path.stem}-train{model_id}'))
            logger.info(f'Starting train {model_id}')
            train(model, data_cnf, model_cnf, train_data=train_data, valid_data=valid_data)
            logger.info(f'Completed train {model_id}')
            
    elif mode == 'cv_train':
        
        all_data = get_data_fn_train(data_cnf=data_cnf)

        kf = KFold(n_splits=5, shuffle=True, random_state=1240)

        for fold, (train_index, val_index) in enumerate(kf.split(all_data)):
            train_subset = [all_data[i] for i in train_index]
            val_subset = [all_data[i] for i in val_index]

            model = Model(DSCA_HLAII, model_path=model_path.with_stem(f'{model_path.stem}-fold{fold}'))
            logger.info(f'Starting Fold {fold}')
            train(model, data_cnf, model_cnf, train_data=train_subset, valid_data=val_subset)
            logger.info(f'Completed Fold {fold}')
            
    elif mode == 'cv_eval':
        mode = 'unseen'
        test_data = get_data_fn_eval(data_cnf=data_cnf, mode=mode)
        
        test_group_name, test_truth = [x[0] for x in test_data], [x[-1] for x in test_data]
        for fold in range(5):
            logger.info(f'Evaluating Fold {fold}')
            model = Model(DSCA_HLAII, model_path=model_path.with_stem(f'{model_path.stem}-fold{fold}'))
            scores_list = []
            scores_list.append(test(model, model_cnf, test_data=test_data))
            result_path = Path(data_cnf['results'])/f'{model_name}-{fold}.txt'
            output_res(test_group_name, test_truth, np.mean(scores_list, axis=0), result_path)
    
    elif mode == 'binding_core':
        pep_seqs, hla_seqs, pep_one_hot, hla_one_hot, pep_esm, hla_esm, cores = get_data_fn_core(data_cnf=data_cnf)

        scores_list = []
        for fold in range(5):
            logger.info(f'Evaluating Fold {fold}')
            model = Model(DSCA_HLAII, model_path=model_path.with_stem(f'{model_path.stem}-fold{fold}'))
            scores = model.predict_core(pep_one_hot, pep_esm, hla_one_hot, hla_esm, core=True)
            scores_list.append(scores)
        
        scores = np.mean(scores_list, axis=0)
        core_positions = scores.argmax(axis=-1)
        print(core_positions)
            
if __name__ == '__main__':
    main()
