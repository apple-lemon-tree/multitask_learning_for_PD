import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import GPT2PreTrainedModel, GPT2LMHeadModel, BertModel, AutoTokenizer
BERT_URL = "/hy-tmp/models/bert-base-uncased"
GPT2_URL = "/hy-tmp/models/gpt2"
class LabelGRU(nn.Module):
    def __init__(self,label_num,input_dim):
        super().__init__()
        self.label_grus = nn.ModuleList(
            [nn.GRU(input_dim, input_dim // 2, batch_first=True, bidirectional=True)
             for _ in range(label_num)]
        )

    def forward(self, inputs, attention_mask, label=None, train=True):
        length = attention_mask.sum(dim=1)
        packed_inputs = pack_padded_sequence(inputs,length.cpu(), batch_first=True, enforce_sorted=False)

        outs = []
        if train:
            for i in label:
                packed_outs2, h_n = self.label_grus[i](packed_inputs)
                out = torch.cat([h_n[-1], h_n[-2]], dim=-1)
                outs.append(out)
        else:
            for gru in self.label_grus:
                packed_outs2, h_n = gru(packed_inputs)
                out = torch.cat([h_n[-1], h_n[-2]], dim=-1)
                outs.append(out)
        return torch.stack(outs,dim=1)

class MyModel(GPT2PreTrainedModel):
    def __init__(self, config, mid_dim, tokenizer, dataset, preseqlen=20,use_adapter=False, content_embeddings=None):
        super().__init__(config)

        self.match_n_layer = config.n_layer
        self.match_n_head = config.n_head
        self.match_n_embd = config.n_embd // config.n_head
        self.n_embed = config.n_embd
        self.preseqlen = preseqlen
        self.mid_dim = mid_dim
        self.dropout = nn.Dropout(0.1)
        self.use_adapter = use_adapter
        self.tokenizer = tokenizer
        self.base_config = config

        if dataset == "Essays":
            self.labels = [
                'extraversion', 'neuroticism','agreeableness','conscientiousness','openness'
            ]
        elif dataset == "Kaggle":
            self.labels = [
                'introversion','extraversion','intuition','sensing','feeling','thinking','perceiving','judgment'
            ]
        self.idx2label = {a: b for a, b in enumerate(self.labels)}
        self.gpt = GPT2LMHeadModel.from_pretrained(GPT2_URL, pad_token_id=tokenizer.pad_token_id)
        self.gpt.resize_token_embeddings(len(self.tokenizer))
        for param in self.gpt.base_model.parameters():
            param.requires_grad = False

        self.bert = BertModel.from_pretrained(BERT_URL)

        for n, param in self.bert.base_model.named_parameters():
            param.requires_grad = False

        self.bert_tokenizer = AutoTokenizer.from_pretrained(BERT_URL)

        self.input_tokens = torch.arange(self.preseqlen).long()
        self.wte = nn.Embedding(self.preseqlen, 768)
        self.content_wte = nn.Embedding.from_pretrained(content_embeddings, freeze=True)

        self.init_label_embedding()
        self.control_trans = nn.Sequential(
            nn.Linear(mid_dim * 2 + 768, self.mid_dim),
            nn.Tanh(),
            nn.Linear(self.mid_dim, config.n_layer * 2 * config.n_embd))

        self.mse_loss = nn.MSELoss()

        self.latent_size = mid_dim
        self.content_hidden2mean = nn.Linear(self.bert.config.hidden_size, mid_dim)
        self.content_hidden2logv = nn.Linear(self.bert.config.hidden_size, mid_dim)
        self.label_hidden2mean = nn.Linear(self.bert.config.hidden_size, mid_dim)
        self.label_hidden2logv = nn.Linear(self.bert.config.hidden_size, mid_dim)
        self.label_gaussian_hidden2mean = nn.Linear(config.n_embd, mid_dim)
        self.content_gaussian_hidden2mean = nn.Linear(config.n_embd, mid_dim)

        self.get_prompt = self.get_prompt_p5
        self.label_gru = LabelGRU(label_num=self.label_embedding.weight.shape[0],
                                  input_dim=self.bert.config.hidden_size)
        self.content_gru = nn.GRU(self.bert.config.hidden_size, self.bert.config.hidden_size // 2,
                                  bidirectional=True, batch_first=True)

    def init_label_embedding(self):
        label_embeddings = []
        for label in self.labels:
            emb = self.bert(**self.bert_tokenizer(label,return_tensors="pt",is_split_into_words=False,
                                                  add_special_tokens=False)).last_hidden_state
            label_embeddings.append(emb.data.mean(1, keepdim=True))
        wte = torch.cat(label_embeddings,dim=1).squeeze()
        self.label_embedding = nn.Embedding.from_pretrained(wte, freeze=True)

    def get_tokens(self, idx_tokens):
        start_idxes = np.multiply(np.asarray(idx_tokens), 10)
        end_idxes = np.multiply(np.asarray(idx_tokens) + 1, 10)
        input_tokens = []
        for s, e in zip(start_idxes, end_idxes):
            input_tokens.append(list(range(s,e)))
        input_tokens = sum(input_tokens, [])
        return input_tokens

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def orthogonal_penalty(self, label_mean, content_mean, l_n_norm=2):
        mm = torch.mm(label_mean, content_mean.permute(1, 0))
        return torch.pow(mm, l_n_norm).mean()

    def get_prompt_p5(
            self,
            batch_size,
            labels,
            bert_ids=None,
            bert_mask=None,
            clusters=None,
            do_train=False
    ):
        if labels is not None:
            labels = labels[0].data.cpu() if len(labels.shape) == 2 else labels
            idx_tokens = [idx for idx, l in enumerate(labels) if l == 1]
            label_num = len(idx_tokens)

            input_tokens = self.input_tokens.expand(batch_size, -1).to(self.device)

            if do_train:
                hidden = self.bert(input_ids=bert_ids, attention_mask=bert_mask).last_hidden_state
                label_v = self.label_gru(hidden, bert_mask, idx_tokens)

                label = bert_mask.sum(dim=1)
                packed_hidden = pack_padded_sequence(hidden, label.cpu(), batch_first=True, enforce_sorted=False)
                packed_hidden, _ = self.content_gru(packed_hidden)
                cls_v, _ = pad_packed_sequence(packed_hidden, batch_first=True, total_length=bert_mask.shape[1])
                cls_v = self.mean_pooling(cls_v[:,:,:], bert_mask[:,:])

                cls_mean = self.content_hidden2mean(cls_v)
                cls_logv = self.content_hidden2logv(cls_v)
                label_mean = self.label_hidden2mean(label_v)
                label_logv = self.label_hidden2logv(label_v)

                content_gauss = self.content_wte(clusters)
                content_mean = self.content_gaussian_hidden2mean(content_gauss)
                label_gauss  = self.label_embedding(torch.tensor(idx_tokens, dtype=torch.long,device=self.device).expand(batch_size,-1))
                label_gauss_mean = self.label_gaussian_hidden2mean(label_gauss)

                q_latent_loss1 = self.mse_loss(content_mean, cls_mean.detach())
                q_latent_loss2 = self.mse_loss(label_gauss_mean, label_mean.detach())
                e_latent_loss1 = self.mse_loss(content_mean.detach(), cls_mean)
                e_latent_loss2 = self.mse_loss(label_gauss_mean.detach(), label_mean)

                content_std = torch.exp(0.5 * cls_logv)
                label_std = torch.exp(0.5 * label_logv)

                z_content = torch.randn([batch_size, self.latent_size], device=self.device)
                z_content = z_content * content_std + cls_mean
                z_content = z_content.unsqueeze(1)

                z_label = torch.randn([batch_size, len(idx_tokens) * 10, self.latent_size], device=self.device)
                z_label = (z_label * label_std.repeat(1, 1, 10).view(label_std.shape[0], 10 * len(idx_tokens), -1) +
                           label_mean.repeat(1, 1, 10).view(label_mean.shape[0], 10 * len(idx_tokens), -1))

                loss_content = 0.2 * q_latent_loss1 + 1 * e_latent_loss1 + torch.mean(content_std) - 1.0 - torch.mean(cls_logv)
                loss_label = 0.2 * q_latent_loss2 + 1 * e_latent_loss2 + torch.mean(label_std) - 1.0 - torch.mean(label_logv)

                loss2 = {"label": loss_label, "content": loss_content}

            else:
                content_gauss = self.content_wte(torch.randint(self.content_wte.weight.shape[0],(batch_size,), device=self.device))
                content_mean = self.content_gaussian_hidden2mean(content_gauss)

                label_gauss = self.label_embedding(torch.tensor(idx_tokens, dtype=torch.long,device=self.device).unsqueeze(0).expand(batch_size,-1))
                label_gauss_mean = self.label_gaussian_hidden2mean(label_gauss)

                z_content = torch.randn([batch_size, self.latent_size], device=self.device)
                z_content = z_content * 1 + content_mean
                z_content = z_content.unsqueeze(1)

                z_label = torch.randn([batch_size, len(idx_tokens) * 10, self.latent_size], device=self.device)

                z_label = z_label * 1 + label_gauss_mean.repeat(1,1,10).view(label_gauss_mean.shape[0], 10 * len(idx_tokens),-1)
                loss2 = None
            temp_control = self.wte(input_tokens)
            temp_control = temp_control.repeat(1,label_num,1)
            temp_control = torch.cat([temp_control, torch.cat([z_label,z_content.repeat(1,z_label.shape[1],1)],dim=-1)],dim=-1)

            past_key_values = self.control_trans(temp_control)
            past_key_values_mlp = past_key_values

            batch_size, seqlen, _ = past_key_values.shape
            past_key_values = past_key_values.view(batch_size, seqlen, self.match_n_layer * 2, self.match_n_head, self.match_n_embd)

            past_key_values = self.dropout(past_key_values)
            past_key_values = past_key_values.permute([2,0,3,1,4]).split(2)
            if do_train:
                return past_key_values,loss2, cls_mean, label_mean
            else:
                return past_key_values,past_key_values_mlp.view(batch_size, seqlen, self.match_n_layer * 2, -1)

    def forward(self,
                input_ids,
                labels,
                label_list,
                bert_ids,
                bert_mask,
                clusters,
                do_train=False,
                **kwargs,
                ):
        batch_size = input_ids.shape[0]
        past_key_values_prompt, loss2, _, _ = self.get_prompt(batch_size=batch_size,labels=label_list,clusters=clusters,
                                                              bert_ids=bert_ids, bert_mask=bert_mask,do_train=do_train)

        past_key_values = past_key_values_prompt

        output = self.gpt(input_ids=input_ids,
                          past_key_values=past_key_values,
                          labels=labels,
                          output_hidden_states=True,
                          **kwargs)

        return output, loss2

class Filter(nn.Module):
    def __init__(self,class_num):
        super().__init__()
        self.bert = BertModel.from_pretrained(BERT_URL)
        self.class_num = class_num
        self.classifier = nn.Linear(768,class_num)

    def forward(self,input_ids,attention_mask):
        hidden_states = self.bert(input_ids=input_ids,attention_mask=attention_mask).last_hidden_state
        cls = hidden_states[:,0,:]
        pred = self.classifier(cls)
        return pred
