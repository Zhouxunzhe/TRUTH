from PIL import Image
import torch
from transformers import AutoProcessor, CLIPVisionModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_id = "bczhou/tiny-llava-v1-hf"
prompt = "USER: <image>\nhow many fingers does the man have? (a) 4 (b) 5 (c) 6 (d) 3\nASSISTANT:"
image_file = "four_finger.jpg"
raw_image = Image.open(image_file)

processor = AutoProcessor.from_pretrained(
    model_id,
    device_map=device
)

inputs = processor(prompt, raw_image, return_tensors='pt').to(0, torch.float16)
print(inputs)

print(CLIPVisionModel(inputs['pixel_values']))
