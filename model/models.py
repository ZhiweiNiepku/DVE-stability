import torch
import torch.nn as nn
import torch.nn.functional as F
import esm
import numpy as np
from .utils import trunc_normal_, Block

class Env(nn.Module):
    # mutual attention weight selection
    def __init__(self):
        super(Env, self).__init__()

    def forward(self, x):
        # --> x[B, H, T] --> [B, T]
        weights = x.mean(1)
        max_inx = torch.argsort(weights, dim=1, descending=True)
        return max_inx  


class ESM_ensamble_cls_env_attention(nn.Module):
    def __init__(self):
        super().__init__() 
        self.esm2, self.esm2_alphabet = esm.pretrained.esm2_t33_650M_UR50D()        

        self.cls_transform = nn.Linear(1280, 1280)
        self.env_transform = nn.Linear(1280, 1280)

        self.cls_classifier = nn.Linear(1280, 1)
        self.cls_const1 = torch.nn.Parameter(torch.ones((1,1280)))
        self.cls_const2 = torch.nn.Parameter(-1 * torch.ones((1,1280)))
        
        self.token_select = Env()
        self.num_token = 32
        self.env_classifier = nn.Linear(1280, 1)
        self.env_const1 = torch.nn.Parameter(torch.ones((1,1280)))
        self.env_const2 = torch.nn.Parameter(-1 * torch.ones((1,1280)))

        self.cls_token = nn.Parameter(torch.zeros(1, 1, 1280))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_token + 1, 1280))
        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.pos_embed, std=.02)

        self.fusion_block = Block(1280, 8)
    
    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}
    
    def forward(self, token_ids1, token_ids2, pos, is_training = False, return_feat = False):       
        """
            token_ids1: [B, L]
            token_ids2: [B, L]
            pos: [B, ]
        """        
        # --> representations, attentions 
        outputs1 = self.esm2.forward(token_ids1, repr_layers=[33], need_head_weights=True)
        outputs2 = self.esm2.forward(token_ids2, repr_layers=[33], need_head_weights=True)
        # --> [B, L+1, 1280]
        outputs1_rep = outputs1['representations'][33]
        outputs2_rep = outputs2['representations'][33]
        
        outputs1_pos_attn = outputs1['attentions'][:, -1]
        outputs2_pos_attn = outputs2['attentions'][:, -1]
        
        outputs1_cls = self.cls_transform(outputs1_rep[:,0,:])
        outputs2_cls = self.cls_transform(outputs2_rep[:,0,:])

        outputs_cls = self.cls_const1 * outputs1_cls + self.cls_const2 * outputs2_cls
        logits_cls = self.cls_classifier(outputs_cls)

        # --> env
        # --> [B, L+1, 1280]
        outputs1_rep_env = self.env_transform(outputs1_rep)
        outputs2_rep_env = self.env_transform(outputs2_rep)
        # --> (B, H, T, T) -> (B, H, T)
        attn1_pos_positions = outputs1_pos_attn[np.arange(len(outputs1_pos_attn)), :, pos].squeeze(dim=2)
        attn2_pos_positions = outputs2_pos_attn[np.arange(len(outputs2_pos_attn)), :, pos].squeeze(dim=2)
        selected_inx1 = self.token_select(attn1_pos_positions)
        selected_inx2 = self.token_select(attn2_pos_positions)
        
        # --> (B, T)
        B = selected_inx1.shape[0]
        selected_tokens1 = [[] for i in range(B)]
        selected_tokens2 = [[] for i in range(B)]
        for i in range(B):
            selected_tokens1[i] = outputs1_rep_env[i, selected_inx1[i,:self.num_token]]
            selected_tokens2[i] = outputs2_rep_env[i, selected_inx2[i,:self.num_token]]
        # --> [B, num_toknes, 1280]
        selected_tokens1 = torch.stack(selected_tokens1)
        selected_tokens2 = torch.stack(selected_tokens2)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        env_tokens1 = torch.cat((cls_tokens, selected_tokens1), dim=1) + self.pos_embed
        env_tokens2 = torch.cat((cls_tokens, selected_tokens2), dim=1) + self.pos_embed

        Env_feat1 = self.fusion_block(env_tokens1)[:, 0]
        Env_feat2 = self.fusion_block(env_tokens2)[:, 0]
        
        outputs_env = self.env_const1 * Env_feat1 + self.env_const2 * Env_feat2
        logits_env = self.env_classifier(outputs_env)

        logtis_mean = (logits_cls + logits_env) / 2
        
        if return_feat:
            return [outputs_cls, outputs_env]

        if is_training:
            return [logits_cls, logits_env, logtis_mean]
        else:
            return logtis_mean