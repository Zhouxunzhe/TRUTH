from PIL import Image
import torch
from torch import nn
from transformers import CLIPConfig, CLIPTextConfig, CLIPVisionConfig
from init import model, processor


class LlavaEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = model.vision_tower
        self.text_config = CLIPTextConfig()
        self.vision_config = CLIPVisionConfig()
        self.config = CLIPConfig.from_text_vision_configs(self.text_config, self.vision_config)

    def forward(self, inputs, vision_feature_layer=None):
        if vision_feature_layer is not None:
            outputs = self.model(inputs, output_hidden_states=True)
            outputs = outputs.hidden_states[vision_feature_layer]
        else:
            outputs = self.model(inputs).last_hidden_state
        return outputs


if __name__ == '__main__':
    clip_config = CLIPConfig()
    encoder = LlavaEncoder()
    img = Image.open('four_finger.jpg')
    prompt = "USER: <image>\nhow many fingers does the man have? (a) 4 (b) 5 (c) 6 (d) 3\nASSISTANT:"
    inputs = processor(prompt, img, return_tensors='pt').to(0, torch.float16)
    out = encoder(inputs['pixel_values'])
    # out = encoder(inputs['pixel_values'], vision_feature_layer=10)
    print(out)
