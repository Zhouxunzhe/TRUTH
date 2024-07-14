from utils import print_trainable_parameters
from generator import internal_contrast_generate, external_contrast_generate
from init import eos_token_id, labels, model_kwargs
import json

# print("model:\n", model)
# print("--------------------------------")
# print("processor:\n", processor)
# print("--------------------------------")
# print_trainable_parameters(model)
# print("--------------------------------")

if __name__ == '__main__':
    alpha = 0.5
    contrast_layer_id = 21
    f = open('data.json', )
    data = json.load(f)
    for data in data['data']:
        image_file1 = 'images/' + data['image_file']
        prompt = data['prompt']
        internal_contrast_generate(image_file1, prompt, alpha, contrast_layer_id,
                                   eos_token_id, labels, 1, model_kwargs)

        # external_contrast_generate(image_file1, prompt, image_file2, alpha,
        #                            eos_token_id, labels, 1, model_kwargs)

        # for i in range(len(outputs.hidden_states)):
        #     logit = decoder.model.lm_head(outputs.hidden_states[i])
        #     print("layer {}:\n{}".format(i, logits2tokens(logits[:, token_size - 1:, :], 10)))
        #     print("--------------------------------")
    f.close()
