from PIL import Image
import torch
from torch import nn
from transformers import LlamaConfig
from init import model, processor
from encoder import LlavaEncoder
from projector import LlavaProjector
from utils import merge_input_ids_with_image_features


class LlavaDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = model.language_model
        self.config = LlamaConfig()

    def forward(self, attention_mask, position_ids, inputs_embeds,
                past_key_values, use_cache, output_attentions, return_dict,
                output_hidden_states):
        outputs = self.model(attention_mask=attention_mask,
                             position_ids=position_ids,
                             past_key_values=past_key_values,
                             output_hidden_states=output_hidden_states,
                             inputs_embeds=inputs_embeds,
                             use_cache=use_cache,
                             output_attentions=output_attentions,
                             return_dict=return_dict,)
        return outputs

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()


if __name__ == '__main__':
    llama_config = LlamaConfig()
    encoder = LlavaEncoder()
    projector = LlavaProjector(model)
    decoder = LlavaDecoder()

    image_file = 'four_finger.jpg'
    prompt = "USER: <image>\nhow many fingers does the man have? (a) 4 (b) 5 (c) 6 (d) 3\nASSISTANT:"
    raw_image = Image.open(image_file)
    inputs = processor(prompt, raw_image, return_tensors='pt').to(0, torch.float16)

    inputs_embeds = decoder.get_input_embeddings()(inputs['input_ids'])
    image_outputs = encoder(inputs['pixel_values'])
    image_features = projector(image_outputs)
    labels = None

    inputs_embeds, attention_mask, labels, position_ids = merge_input_ids_with_image_features(
        model, image_features, inputs_embeds, inputs['input_ids'], inputs['attention_mask'], labels
    )
    if labels is None:
        labels = torch.full_like(attention_mask, model.config.ignore_index).to(torch.long)
    out = decoder(
        attention_mask=attention_mask,
        position_ids=position_ids,
        inputs_embeds=inputs_embeds,
        output_hidden_states=True,
        past_key_values=None,
        use_cache=True,
        output_attentions=False,
        return_dict=True,
    )
    print(out)
