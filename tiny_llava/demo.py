from PIL import Image
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration

model_id = "bczhou/tiny-llava-v1-hf"
image_file = 'images/four_finger.jpg'
prompt = "USER: <image>\nhow many fingers does the man have? (a) 4 (b) 5 (c) 6 (d) 3\nASSISTANT:"
model = LlavaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    ).to(0)
processor = AutoProcessor.from_pretrained(model_id)
raw_image = Image.open(image_file)
inputs = processor(prompt, raw_image, return_tensors='pt').to(0, torch.float16)
output = model.generate(**inputs, max_new_tokens=200, do_sample=False)
print(output.size())
print(processor.decode(output[0], skip_special_tokens=True))
