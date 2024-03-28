import torch
import torch.distributed as dist
from transformers import AutoProcessor, AutoTokenizer, \
    LlavaForConditionalGeneration
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import StoppingCriteriaList


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = 'cpu'
print("Using device: ", device)

# model_id = "bczhou/TinyLLaVA-1.5B"
model_id = "bczhou/tiny-llava-v1-hf"
model = LlavaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    # load_in_4bit=True,
).to(0)
processor = AutoProcessor.from_pretrained(
    model_id
)
tokenizer = AutoTokenizer.from_pretrained(model_id)
logits_processor = LogitsProcessorList()
stopping_criteria = StoppingCriteriaList()

synced_gpus = None
if synced_gpus is None:
    if is_deepspeed_zero3_enabled() and dist.get_world_size() > 1:
        synced_gpus = True
    else:
        synced_gpus = False

model._validate_model_class()
generation_config = model.generation_config

max_length = None
labels = None
generation_config.max_new_tokens = 200
model_kwargs = generation_config.update()
generation_config.validate()
model._validate_model_kwargs(model_kwargs.copy())
pad_token_id = model.generation_config.pad_token_id
eos_token_id = model.generation_config.eos_token_id

if generation_config.pad_token_id is None and generation_config.eos_token_id is not None:
    eos_token_id = generation_config.eos_token_id
    if isinstance(eos_token_id, list):
        eos_token_id = eos_token_id[0]
    generation_config.pad_token_id = eos_token_id
