import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class GetEmbeddings(nn.Module):
    def __init__(self, model_url, extractor):
        super().__init__()
        self.embedding = AutoModel.from_pretrained(model_url,trust_remote_code=True)
        self.extractor = extractor
    def forward(self, input_ids, attention_mask,texts):
        semantic_embeddings = self.embedding(input_ids=input_ids, attention_mask=attention_mask)
        senticnet_embeddings = self.extractor.extract_features_batch(texts=texts)
        return semantic_embeddings.last_hidden_state, torch.from_numpy(senticnet_embeddings).to("cuda")


class MultiDilatedConvBank(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_sizes=(3,4,5), dilations=(1,2)):
        super().__init__()
        self.convs = nn.ModuleList()
        for k in kernel_sizes:
            for r in dilations:
                # 多尺度膨胀卷积核
                self.convs.append(nn.Conv1d(in_ch, out_ch, kernel_size=k, dilation=r, padding=(k-1)*r//2))
        self.out_dim = out_ch * len(self.convs)  # 128 * 6 = 768

    def forward(self, x):  # x: [8, seq_len、768, 768]
        x = x.permute(0,2,1)  # -> [B, 768, seq_len、768]
        outs = []
        for conv in self.convs:
            o = conv(x)                # [B, 128, L]
            o = F.relu(o)
            p = F.max_pool1d(o, kernel_size=o.size(2)).squeeze(2)  # [B, 128]
            outs.append(p)
        return torch.cat(outs, dim=1)  # [B, out_ch * n_scales]

class TokenConvMap(nn.Module):
    """Produce token-wise key/value maps from H for attention"""
    def __init__(self, in_dim, mid_dim):
        super().__init__()
        self.proj_k = nn.Linear(in_dim, mid_dim)
        self.proj_v = nn.Linear(in_dim, mid_dim)

    def forward(self, H):  # H: [8, 768, 768]
        K = self.proj_k(H)  # [B, L, d_k]
        V = self.proj_v(H)  # [B, L, d_k]
        return K, V

class GatedFusion(nn.Module):
    def __init__(self, input_dim, hidden=768):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.Sigmoid()
        )

    def forward(self, semantic, emotional):
        # semantic & emotional: [8, 768]
        z = torch.cat([semantic, emotional], dim=1) # [8, 1536]
        g = self.gate(z)  # [8, 768]
        return g * semantic + (1 - g) * emotional

class HMD_GCF_Module(nn.Module):
    def __init__(self, bert_dim=768, sentic_dim=100, conv_out=128,
                 kernel_sizes=(2,3,4), dilations=(1,2), att_dim=128, dropout=0.3):
        super().__init__()
        # 多处都卷积提取局部语义模式
        self.conv_bank = MultiDilatedConvBank(in_ch=bert_dim, out_ch=conv_out,
                                              kernel_sizes=kernel_sizes, dilations=dilations)
        # token映射层，供Attention使用
        self.token_map = TokenConvMap(in_dim=bert_dim, mid_dim=att_dim)
        # 将情感特征映射为Query向量
        self.q_proj = nn.Linear(sentic_dim, att_dim)
        self.att_scale = att_dim ** 0.5  # 缩放因子
        # 池化卷积特征投影
        self.pool_proj = nn.Linear(self.conv_bank.out_dim, att_dim) # nn.Linear(768, 128)
        # 将注意力输出的情感映射到语义维度
        self.emotion_pool = nn.Sequential(
            nn.Linear(att_dim, 448),  # nn.Linear(128, 448)
            nn.ReLU(),
            nn.Linear(448, self.conv_bank.out_dim), # nn.Linear(448, 768)
        )
        # 门控融合模块
        self.gated = GatedFusion(input_dim=2*self.conv_bank.out_dim)
        # final MLP shared
        self.shared_fc = nn.Sequential(
            # nn.Linear(self.conv_bank.out_dim, 512),
            # nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, H, sentic_vec, cls_pool=None):
        """
        H: [B, L, d] BERT token embeddings
        sentic_vec: [B, c]  SenticNet / AffectSpace
        cls_pool: optional [B, d] e.g., BERT pooled output
        returns: shared features [B, D]
        """
        B = H.size(0)
        # 1) 多尺度卷积特征
        U = self.conv_bank(H)  # [8, 768]
        # 2) Token级别的键值对
        K, V = self.token_map(H)  # [8, 768, 128]
        # 3) 将情感向量映射为Query
        Q = self.q_proj(sentic_vec).unsqueeze(1)  # [B, 1, att_dim]
        # 4) 交叉注意力权重
        # K: [8,768,128] -> transpose to [8, 128, 768] for matmul or use bmm
        scores = torch.bmm(Q, K.transpose(1,2)) / self.att_scale  # [B,1,L]
        A = torch.softmax(scores, dim=-1)  # [B,1,L]
        # 5) 加权求和得到情感引导摘要
        C = torch.bmm(A, V).squeeze(1)  # [8, 128]
        # 将情感向量扩大到语义向量的维度
        emotional = self.emotion_pool(C)  # [8, 768] (D == U dim)
        # Optionally include cls_pool: concat in gate input (here omitted for simplicity)
        # 5) 门控融合
        fused = self.gated(U, emotional)  # [B, 768]
        out = self.shared_fc(fused)  # [B, 768]
        return out


class ClassficationHead(nn.Module):
    def __init__(self, personality_type,emotion_type):
        super().__init__()
        label_num = {
            'essays': 5,
            "myPersonality": 5,
            'mbti': 4,
            'isear': 7,
            'tec': 6
        }
        self.pnd_head = nn.Linear(768,label_num[personality_type])
        self.emo_head = nn.Linear(768,label_num[emotion_type])

    def forward(self, out,data_type):
        if data_type in ('essays','myPersonality','mbti'):
            output = self.pnd_head(out)
        elif data_type in ('isear','tec'):
            output = self.emo_head(out)
        return output


