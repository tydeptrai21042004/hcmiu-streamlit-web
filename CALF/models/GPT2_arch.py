import torch
from transformers.models.gpt2.modeling_gpt2 import GPT2Model


class AccustumGPT2Model(GPT2Model):
    """Small CALF wrapper around HuggingFace GPT2Model.

    CALF calls the GPT-2 branch as:
        final_feat, intermediate_feat = self.gpt2(inputs_embeds=...)

    HuggingFace GPT2Model already supports inputs_embeds and can return
    hidden_states when output_hidden_states=True. This wrapper keeps the
    CALF calling convention while avoiding version-specific copied GPT-2
    internals that often break across transformers versions.
    """

    def forward(self, input_ids=None, labels=None, **kwargs):
        outputs = super().forward(input_ids=input_ids, **kwargs)
        return outputs.last_hidden_state, outputs.hidden_states
