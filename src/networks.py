#!/usr/bin/env python3
# -*- coding: utf-8
"""
Created on 2025/09/18
@author yhj

"""

# %%
import math
import torch
from torch import nn
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore")

# %%
class Self_Attention(nn.Module):
    def __init__(self,input_dim,dim_k,dim_v):
        super(Self_Attention,self).__init__()
        self.q = nn.Linear(input_dim,dim_k)
        self.k = nn.Linear(input_dim,dim_k)
        self.v = nn.Linear(input_dim,dim_v)
        self._norm_fact = 1 / math.sqrt(dim_k)
        
    
    def forward(self,x):
        Q = self.q(x) # Q: batch_size * seq_len * dim_q
        K = self.k(x) # K: batch_size * seq_len * dim_k
        V = self.v(x) # V: batch_size * seq_len * dim_v
         
        atten = nn.Softmax(dim=-1)(torch.bmm(Q,K.permute(0,2,1))) * self._norm_fact # Q * K.T() # batch_size * seq_len * seq_len
        
        output = torch.bmm(atten,V) # Q * K.T() * V # batch_size * seq_len * dim_v
        
        return output

class CNN(nn.Module):
    def __init__(self, input_dim, out_dim, kernel):
        super(CNN,self).__init__()
        self.conv1d = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, 
                      out_channels=out_dim, 
                      kernel_size=kernel, 
                      stride=1, padding='same', 
                      bias=True),
            nn.BatchNorm1d(out_dim),
            nn.LeakyReLU(0.02)
        )
        
    def forward(self, x):
        output = self.conv1d(x)
        return output

class TextCNN(nn.Module):
    def __init__(self, input_dim, out_dim, kernel=[]):
        super(TextCNN,self).__init__()
        layer = []
        for i,os in enumerate(kernel):
            layer.append(CNN(input_dim, out_dim, os))
        self.layer = nn.ModuleList(layer)
    def forward(self, x):
        x = x.permute(0, 2, 1)
        o1 = self.layer[0](x).permute(0, 2, 1)
        o2 = self.layer[1](x).permute(0, 2, 1)
        o3 = self.layer[2](x).permute(0, 2, 1)
        o4 = self.layer[3](x).permute(0, 2, 1)
        return o1, o2, o3, o4

class ConvNN(nn.Module):
    def __init__(self,in_dim,c_dim,kernel_size):
        super(ConvNN,self).__init__()
        self.convs = nn.Sequential(
            nn.Conv1d(in_channels=in_dim, out_channels= c_dim, kernel_size=kernel_size,padding='same'),
            nn.ReLU(),
            nn.Conv1d(in_channels=c_dim, out_channels= c_dim*2, kernel_size=kernel_size,padding='same'),
            nn.ReLU(),
            nn.Conv1d(in_channels=c_dim*2, out_channels= c_dim*3, kernel_size=kernel_size,padding='same'),
            nn.ReLU(),
            )
    def forward(self,x):
        x = self.convs(x)
        return x

class res(nn.Module):
    def __init__(self, dim, size):
        super().__init__()
        self.dim = dim
        self.size = size
        self.downsample = nn.Conv1d(dim, dim, size)

    def forward(self, x):
        x = self.downsample(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x

class CrossAttention(nn.Module):
    def __init__(self, dim):
        super(CrossAttention, self).__init__()
        self.dim = dim
        self.q1 = nn.Linear(dim, dim)
        self.k1 = nn.Linear(dim, dim)
        self.v1 = nn.Linear(dim, dim)

    def forward(self, x, x_res, y_res):
        q1 = self.q1(x)
        k1 = self.k1(x_res)
        v1 = self.v1(y_res)

        attn1 = F.softmax(torch.matmul(q1, k1.transpose(1, 2)) / (self.dim ** 0.5), dim=-1)
        out = torch.einsum('bij,bkv->biv', attn1, v1)
        out = torch.nn.functional.normalize(out, p=2, dim=2)
        return out + x 

class BasicTransformerStage(nn.Module):
    def __init__(self, dim, depth, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=num_heads,
                dim_feedforward=int(dim * mlp_ratio),
                dropout=dropout,
                batch_first=True
            )
            for _ in range(depth)
        ])

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x
    
class GlobalMaxPool1d(nn.Module):
    def __init__(self):
        super(GlobalMaxPool1d,self).__init__()
    def forward(self,x):
        output, _ = torch.max(x,1)
        return output

# %%
class DSCA_HLAII(nn.Module):
    def __init__(self, **kwargs):
        super(DSCA_HLAII, self).__init__()

        self.embed_seq = nn.Linear(22,128)
        self.dense_esm_pep = nn.Linear(1152,128)
        self.dense_esm_hla = nn.Linear(1152,128)
        
        self.global_max_pooling = GlobalMaxPool1d()
        self.pep_residue = nn.Sequential(nn.Linear(192,1), nn.Sigmoid())

        self.stream_pep_o_res1 = res(384, 9)
        self.stream_hla_o_res1 = res(384, 5)
        self.conv_pep = res(384, 9)
        self.conv_hla = res(384, 5)
        
        self.pep_transformerblock1 = BasicTransformerStage(dim=384, depth=3, num_heads=4)
        self.hla_transformerblock1 = BasicTransformerStage(dim=384, depth=3, num_heads=4)
        
        self.cross_attention_pep = CrossAttention(384)
        self.cross_attention_hla = CrossAttention(384)
        
        self.dnns = nn.Sequential(
            nn.Linear(768,1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024,1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024,512))
        
        self.att = Self_Attention(128,128,128)
        self.output = nn.Linear(512,1)

    def forward(self, x_pep, esm_x_pep, x_hla, esm_x_hla, core=False, **kwargs):
        # ================= Feature embedding =================
        esm_x_hla = torch.cat([esm_x_hla[:, 0], esm_x_hla[:, 1]], dim=1)

        pep_seq_emb = self.embed_seq(x_pep)
        hla_seq_emb = self.embed_seq(x_hla)
        pep_esm_emb = self.dense_esm_pep(esm_x_pep)
        hla_esm_emb = self.dense_esm_hla(esm_x_hla)
        
        # =================Feature concatenate=================
        encode_peptide = torch.cat([pep_seq_emb, pep_esm_emb],dim=-1)
        encode_hla = torch.cat([hla_seq_emb, hla_esm_emb],dim=-1)

        # =================Self-Attention model=================
        # -------------------For protein-------------------
        prot_seq_att = self.embed_seq(x_hla)
        protein_att = self.att(prot_seq_att)
        # -------------------For peptide-------------------
        pep_seq_att = self.embed_seq(x_pep)
        peptide_att = self.att(pep_seq_att)

        # =================Cross-Attention model=================
        feature_pep_1 = torch.cat([encode_peptide, peptide_att], dim=-1)
        feature_p_1 = torch.cat([encode_hla, protein_att], dim=-1)
        
        pep1_o_res = self.stream_pep_o_res1(feature_pep_1)
        hla1_o_res = self.stream_hla_o_res1(feature_p_1)
        
        feature_pep_2 = self.pep_transformerblock1(feature_pep_1)
        feature_p_2 = self.hla_transformerblock1(feature_p_1)

        feature_pep = self.cross_attention_pep(feature_pep_2, pep1_o_res, hla1_o_res)
        feature_p = self.cross_attention_hla(feature_p_2, hla1_o_res, pep1_o_res)

        feature_pep = self.conv_pep(feature_pep)
        feature_p = self.conv_hla(feature_p)
        
        if core == False:
        # =================Global max pooling=================
            glomax_pep = self.global_max_pooling(feature_pep)
            glomax_p = self.global_max_pooling(feature_p)

            # =================Feature concatenate=================
            encode_interaction = torch.cat([glomax_pep, glomax_p], dim=-1)
    
            # =================        MLP        =================
            encode_interaction = self.dnns(encode_interaction)
            predictions = torch.sigmoid(self.output(encode_interaction))
            return predictions
        
        elif core == True:
            predictions = torch.mean(feature_pep, dim=-1)
            return predictions
