import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from AttentionMechanism import AttentionMechanism
from networks.module import VisionTransformer, Attention

class CrossAttention(nn.Module):
    def __init__(self, in_dim1, in_dim2, config):
        super(CrossAttention, self).__init__()
        self.num_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_heads)
        self.all_head_size = self.num_heads * self.attention_head_size
        self.num_heads = config.transformer["num_heads"]
        
        # projection
        self.proj_q1 = nn.Linear(in_dim1, self.all_head_size, bias=False)
        self.proj_k2 = nn.Linear(in_dim2, self.all_head_size, bias=False)
        self.proj_v2 = nn.Linear(in_dim2, self.all_head_size, bias=False)
        self.proj_o = nn.Linear(self.all_head_size, in_dim1)
        
    def forward(self, x1, x2, mask=None):
        x1 = x1.flatten(2)
        x1 = x1.transpose(-1, -2)  # (B, n_patches, hidden)
        x2 = x2.flatten(2)
        x2 = x2.transpose(-1, -2)  # (B, n_patches, hidden)

        batch_size, seq_len1, in_dim1 = x1.size()
        seq_len2 = x2.size(1)
        
        q1 = self.proj_q1(x1).view(batch_size, seq_len1, self.num_heads, self.attention_head_size).permute(0, 2, 1, 3)
        k2 = self.proj_k2(x2).view(batch_size, seq_len2, self.num_heads, self.attention_head_size).permute(0, 2, 3, 1)
        v2 = self.proj_v2(x2).view(batch_size, seq_len2, self.num_heads, self.attention_head_size).permute(0, 2, 1, 3)
        
        attn = torch.matmul(q1, k2) / self.attention_head_size**0.5
        
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        
        attn = F.softmax(attn, dim=-1)
        output = torch.matmul(attn, v2).permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len1, -1)
        output = self.proj_o(output)
        output = output.permute(0, 2, 1)
        h, w = int(np.sqrt(seq_len1)), int(np.sqrt(seq_len1))
        output = output.contiguous().view(batch_size, in_dim1, h, w)
        return output

class net_multi_Transformer(nn.Module):
    def __init__(self,config,backbone1,backbone2,img_size=224,num_classes=17,zero_head=False,vis=False):
        super(net_multi_Transformer, self).__init__()
        self.low_channel_Transformer = VisionTransformer(config, backbone1, img_size, num_classes, zero_head, vis)
        self.high_channel_Transformer = VisionTransformer(config, backbone2, img_size, num_classes, zero_head, vis)
        self.low_channel_cross_attention = AttentionMechanism(num_classes)
        self.high_channel_cross_attention = AttentionMechanism(num_classes)
        self.low_channel_Transformer.load_from(weights=np.load(config.pretrained_path))
        self.high_channel_Transformer.load_from(weights=np.load(config.pretrained_path))

    def forward(self, x):
        low_outputs1,low_outputs2  = self.low_channel_Transformer(x[:,:4,:,:])
        high_outputs1, high_outputs2 = self.high_channel_Transformer(x[:, 4:, :, :])
        logits1 = patch_attention(low_outputs1, high_outputs1, self.low_channel_cross_attention, patch_size=64)
        logits2 = patch_attention(low_outputs2, high_outputs2, self.high_channel_cross_attention, patch_size=64)
        return logits1,logits2 #

def patch_attention(x1, x2, model, patch_size=64):
    batch_size, c, h, w = x1.size()
    num_patches_h = h // patch_size
    num_patches_w = w // patch_size

    results = torch.zeros_like(x1).to(x1.device)

    for i in range(num_patches_h):
        for j in range(num_patches_w):
            start_h, end_h = i * patch_size, (i + 1) * patch_size
            start_w, end_w = j * patch_size, (j + 1) * patch_size

            patch_x1 = x1[:, :, start_h:end_h, start_w:end_w]
            patch_x2 = x2[:, :, start_h:end_h, start_w:end_w]

            patch_out = model(patch_x1, patch_x2)

            results[:, :, start_h:end_h, start_w:end_w] = patch_out

    return results

if __name__ == "__main__":
    pass