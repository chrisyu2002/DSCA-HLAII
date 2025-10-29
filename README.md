# DSCA-HLAII

The interaction between peptides and human leukocyte antigen class II (HLA-II) molecules plays a pivotal role in adaptive immune responses, as HLA-II mediates the recognition of exogenous antigens and initiates T cell activation through peptide presentation. Accurate prediction of peptide-HLA-II binding serves as a cornerstone for deciphering cellular immune responses, and is essential for guiding the optimization of antibody therapeutics. Researchers have developed several computational approaches to identify peptide-HLA-II interaction and presentation. However, most computational approaches exhibit inconsistent predictive performance, poor generalization ability and limited biological interpretability. **We propose DSCA-HLAII, an end-to-end deep learning framework based on a Dual-Stream Cross-Attention (DSCA) mechanism, designed to provide a robust and high-performance solution for predicting peptide–HLA-II presentation probabilities.**



**Given the complexity and instability of individuals in configuring the environment, we strongly recommend that users use DSCA-HLAII's online prediction Web server, which can be accessed through http://bliulab.net/DSCA-HLAII/.**



![DSCA-HLAII](/imgs/DSCA-HLAII.png)

**Fig. 1: Overview of the DSCA-HLAII framework. A** Data preparation workflow. **B** The Residue-level Embedding module. This module extracts ONE-HOT and ESMC representations while incorporating context-enhanced embeddings, providing a comprehensive representation of both peptides and HLA-II molecules. **C** The Representation Extraction module. This module captures multi-level dependencies in sequences from global and local perspectives. **D** The Cross Attention module. This module employs the DSCA mechanism to capture the information interaction between peptides and HLA-II molecules. **E** The Presentation Prediction module. This module outputs the predicted presentation probability based on the integrated interaction features of peptides and HLA-II molecules. **F** Downstream Tasks. DSCA-HLAII is used for predicting peptide binding cores and assessing antibody immunogenicity.



# 1 Installation



## 1.1 Create conda environment

```

conda create -n DSCA-HLAII python=3.10

conda activate DSCA-HLAII

```



## 1.2 Requirements

We recommend installing the environment using the provided `environment.yaml` file to ensure compatibility:

```

conda env update -f environment.yaml

```

If this approach fails or Conda is not available, you can manually install the main dependencies as listed below:

```

python  3.10

numpy 1.26.2

pandas 2.2.3

scikit-learn 1.6.1

tokenizers 0.15.0

torch 2.6.0+cu118

torchaudio 2.6.0+cu118

tqdm 4.67.1

transformers 4.52.1

logzero 1.7.0

ruamel-yaml 0.18.10

click 8.1.8

```



## 1.3 Tools

A large protein language model are required: 

```

ESM C(600M) \\

```

How to install ESM C(600M):

Download (More information, please see **[SPYfighting/esm-C (github.com)](https://github.com/SPYfighting/esm-C)**)

```

pip install esm

```



## 1.4 Install DSCA-HLAII

```

git clone git@github.com:chrisyu2002/DSCA-HLAII.git

```



Besides, due to the file size limitation of Git LFS, the pre-trained (ESM C) feature files of peptide sequences and HLA-II sequences used for training and testing in DSCA-HLAII are available through the **https://pan.quark.cn/s/5e60ba8f95a0** (about 1.1TB). However, this does not affect the testing of the model.

```

mv esm_3_pep_train_1_to_20w.npy data/pep/

mv esm_3_pep_train_20_to_40w.npy data/pep/

......

```





**Finally, configure the Defalut path of the above tool and the database in `data.yaml`. You can change the path of the tool and database by configuring `data.yaml` as needed.**





# 2 Usage

The procedure for predicting peptide–HLA-II presentation probability and binding cores is as follows:



(1) We provide five operational modes: `'input'`, `'train'`, `'cv_train'`, `'cv_eval'`, and `'binding_core'`. The default mode is `'input'`, which predicts the binding cores between peptides and HLA-II molecules in the input file.



(2) The format of the input file can be referenced from our default example file. The default file path can be modified in `data.yaml`, and you may replace it with your own file for prediction.



(3) Run `main.py` to perform the prediction.

```

python main.py

```



(4) The output files will be saved in the `results` directory.



If you want to retrain based on your private dataset, find the original DSCA-HLAII model in `src/networks.py`. The DSCA-HLAII source code we wrote is based on the Pytorch implementation and can be easily imported by instantiating it.







# 3 Problem feedback

If you have questions on how to use DSCA-HLAII, feel free to raise questions in the \[discussions section](https://github.com/chrisyu2002/DSCA-HLAII/discussions). If you identify any potential bugs, feel free to raise them in the \[issuetracker](https://github.com/chrisyu2002/DSCA-HLAII/issues).



In addition, if you have any further questions about DSCA-HLAII, please feel free to contact us [**hjyu@bliulab.net**]



\# 4 Citation



If you find our work useful, please cite us at

```

@unpublished{yan2025dsca,
  title={DSCA-HLAII: A Dual-Stream Cross-Attention Model for Predicting Peptide--HLA Class II Interactions and Presentation},
  author={Ke Yan and Hongjun Yu and Shutao Chen and Bin Liu},
  note={Manuscript in preparation}
}

```



