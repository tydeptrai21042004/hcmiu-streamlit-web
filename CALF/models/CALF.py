import torch
import torch.nn as nn
from einops import rearrange
from peft import LoraConfig, TaskType, get_peft_model
from models.GPT2_arch import AccustumGPT2Model


class Encoder_PCA(nn.Module):
    def __init__(self, input_dim, word_embedding, hidden_dim=768, num_heads=12, num_encoder_layers=1):
        super(Encoder_PCA, self).__init__()
        self.linear = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.cross_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads)
        self.register_buffer("word_embedding_base", word_embedding.T.detach().clone().float())

    def forward(self, x):
        B = x.shape[0]
        word_embedding = self.word_embedding_base
        if word_embedding.ndim == 2:
            word_embedding = word_embedding.unsqueeze(0).repeat(B, 1, 1)
        elif word_embedding.shape[0] != B:
            word_embedding = word_embedding[:1].repeat(B, 1, 1)

        x = self.linear(x)
        x = self.transformer_encoder(x.transpose(0, 1)).transpose(0, 1)
        x_time = x

        q = x.transpose(0, 1)
        k = v = word_embedding.transpose(0, 1)
        x_text, _ = self.cross_attention(q, k, v)
        x_text = x_text.transpose(0, 1)
        return x_time, x_text


class Model(nn.Module):
    def __init__(self, configs, device):
        super(Model, self).__init__()
        self.pred_len = configs.pred_len
        self.task_name = configs.task_name

        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=configs.r,
            lora_alpha=configs.lora_alpha,
            lora_dropout=configs.lora_dropout,
            target_modules=["c_attn"],
        )

        self.gpt2 = AccustumGPT2Model.from_pretrained(
            "gpt2", output_attentions=True, output_hidden_states=True
        )
        self.gpt2_text = AccustumGPT2Model.from_pretrained(
            "gpt2", output_attentions=True, output_hidden_states=True
        )

        self.gpt2.h = self.gpt2.h[:configs.gpt_layers]
        self.gpt2_text.h = self.gpt2_text.h[:configs.gpt_layers]
        self.gpt2 = get_peft_model(self.gpt2, peft_config)

        word_embedding = torch.load(configs.word_embedding_path, map_location=device)
        if not torch.is_tensor(word_embedding):
            word_embedding = torch.tensor(word_embedding)
        word_embedding = word_embedding.float().to(device=device)

        for _, param in self.gpt2.named_parameters():
            param.requires_grad = False
        for name, param in self.gpt2.named_parameters():
            if "ln" in name or "wpe" in name or "lora" in name:
                param.requires_grad = True

        for _, param in self.gpt2_text.named_parameters():
            param.requires_grad = False
        for name, param in self.gpt2_text.named_parameters():
            if "wpe" in name:
                param.requires_grad = True

        self.time_proj = nn.ModuleList(
            [nn.Linear(configs.d_model, configs.d_model, bias=False) for _ in range(configs.gpt_layers + 1)]
        )
        self.text_proj = nn.ModuleList(
            [nn.Linear(configs.d_model, configs.d_model, bias=False) for _ in range(configs.gpt_layers + 1)]
        )
        self.in_layer = Encoder_PCA(configs.seq_len, word_embedding, hidden_dim=configs.d_model)

        if self.task_name in ["long_term_forecast", "short_term_forecast"]:
            self.out_layer = nn.Linear(configs.d_model, configs.pred_len)
        elif self.task_name == "classification":
            self.out_layer = nn.Linear(configs.d_model * configs.enc_in, configs.num_class)
        elif self.task_name == "imputation":
            self.out_layer = nn.Linear(configs.d_model, configs.seq_len)
        elif self.task_name == "anomaly_detection":
            self.out_layer = nn.Linear(configs.d_model, configs.seq_len)
        else:
            raise ValueError(f"Unsupported task_name: {self.task_name}")

        for layer in (self.gpt2_text, self.gpt2, self.in_layer, self.out_layer, self.time_proj, self.text_proj):
            layer.to(device=device)
            layer.train()

    def _project_hidden_states(self, hidden_states, proj_layers):
        hidden_states = list(hidden_states) if hidden_states is not None else []
        if len(hidden_states) == 0:
            return tuple()
        n = min(len(hidden_states), len(proj_layers))
        return tuple(proj_layers[idx](hidden_states[idx]) for idx in range(n))

    def forecast(self, x):
        B, L, M = x.shape
        means = x.mean(1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x = x / stdev

        x = rearrange(x, "b l m -> b m l")
        outputs_time1, outputs_text1 = self.in_layer(x)

        outputs_time, intermediate_feat_time = self.gpt2(inputs_embeds=outputs_time1)
        outputs_text, intermediate_feat_text = self.gpt2_text(inputs_embeds=outputs_text1)

        outputs_time = outputs_time + outputs_time1
        outputs_text = outputs_text + outputs_text1

        intermediate_feat_time = self._project_hidden_states(intermediate_feat_time, self.time_proj)
        intermediate_feat_text = self._project_hidden_states(intermediate_feat_text, self.text_proj)

        outputs_time = self.out_layer(outputs_time[:, -M:, :])
        outputs_text = self.out_layer(outputs_text[:, -M:, :])

        outputs_time = rearrange(outputs_time, "b m l -> b l m")
        outputs_text = rearrange(outputs_text, "b m l -> b l m")

        outputs_text = outputs_text * stdev + means
        outputs_time = outputs_time * stdev + means

        return {
            "outputs_text": outputs_text,
            "outputs_time": outputs_time,
            "intermidiate_time": intermediate_feat_time,
            "intermidiate_text": intermediate_feat_text,
        }

    def classification(self, x):
        B, L, M = x.shape
        x = rearrange(x, "b l m -> b m l")
        outputs_time1, outputs_text1 = self.in_layer(x)
        outputs_time, intermediate_feat_time = self.gpt2(inputs_embeds=outputs_time1)
        outputs_text, intermediate_feat_text = self.gpt2_text(inputs_embeds=outputs_text1)

        outputs_time = outputs_time + outputs_time1
        outputs_text = outputs_text + outputs_text1
        intermediate_feat_time = self._project_hidden_states(intermediate_feat_time, self.time_proj)
        intermediate_feat_text = self._project_hidden_states(intermediate_feat_text, self.text_proj)

        outputs_time = self.out_layer(outputs_time.reshape(B, -1))
        outputs_text = self.out_layer(outputs_text.reshape(B, -1))
        return {
            "outputs_text": outputs_text,
            "outputs_time": outputs_time,
            "intermidiate_time": intermediate_feat_time,
            "intermidiate_text": intermediate_feat_text,
        }

    def imputation(self, x, mask):
        B, L, M = x.shape
        means = x.mean(1, keepdim=True).detach()
        x = x - means
        x = x.masked_fill(mask == 0, 0)
        stdev = torch.sqrt(torch.sum(x ** 2, dim=1) / torch.sum(mask == 1, dim=1) + 1e-5).unsqueeze(1).detach()
        x = x / stdev

        x = rearrange(x, "b l m -> b m l")
        outputs_time1, outputs_text1 = self.in_layer(x)
        outputs_time, intermediate_feat_time = self.gpt2(inputs_embeds=outputs_time1)
        outputs_text, intermediate_feat_text = self.gpt2_text(inputs_embeds=outputs_text1)

        outputs_time = outputs_time + outputs_time1
        outputs_text = outputs_text + outputs_text1
        intermediate_feat_time = self._project_hidden_states(intermediate_feat_time, self.time_proj)
        intermediate_feat_text = self._project_hidden_states(intermediate_feat_text, self.text_proj)

        outputs_time = rearrange(self.out_layer(outputs_time), "b m l -> b l m")
        outputs_text = rearrange(self.out_layer(outputs_text), "b m l -> b l m")
        outputs_text = outputs_text * stdev + means
        outputs_time = outputs_time * stdev + means
        return {
            "outputs_text": outputs_text,
            "outputs_time": outputs_time,
            "intermidiate_time": intermediate_feat_time,
            "intermidiate_text": intermediate_feat_text,
        }

    def anomaly_detection(self, x):
        B, L, M = x.shape
        means = x.mean(1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x = x / stdev

        x = rearrange(x, "b l m -> b m l")
        outputs_time1, outputs_text1 = self.in_layer(x)
        outputs_time, intermediate_feat_time = self.gpt2(inputs_embeds=outputs_time1)
        outputs_text, intermediate_feat_text = self.gpt2_text(inputs_embeds=outputs_text1)

        outputs_time = outputs_time + outputs_time1
        outputs_text = outputs_text + outputs_text1
        intermediate_feat_time = self._project_hidden_states(intermediate_feat_time, self.time_proj)
        intermediate_feat_text = self._project_hidden_states(intermediate_feat_text, self.text_proj)

        outputs_time = rearrange(self.out_layer(outputs_time), "b m l -> b l m")
        outputs_text = rearrange(self.out_layer(outputs_text), "b m l -> b l m")
        outputs_text = outputs_text * stdev + means
        outputs_time = outputs_time * stdev + means
        return {
            "outputs_text": outputs_text,
            "outputs_time": outputs_time,
            "intermidiate_time": intermediate_feat_time,
            "intermidiate_text": intermediate_feat_text,
        }

    def forward(self, x, mask=None):
        if self.task_name in ["long_term_forecast", "short_term_forecast"]:
            return self.forecast(x)
        if self.task_name == "classification":
            return self.classification(x)
        if self.task_name == "imputation":
            return self.imputation(x, mask)
        if self.task_name == "anomaly_detection":
            return self.anomaly_detection(x)
        raise ValueError(f"Unsupported task_name: {self.task_name}")
