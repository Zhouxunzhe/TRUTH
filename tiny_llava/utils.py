import torch
from init import tokenizer


class Config:
    def __init__(self, output_hidden_states=True, past_key_values=None,
                 use_cache=True, output_attentions=False, return_dict=True, ):
        self.output_hidden_states = output_hidden_states
        self.past_key_values = past_key_values
        self.use_cache = use_cache
        self.output_attentions = output_attentions
        self.return_dict = return_dict


def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )


def merge_input_ids_with_image_features(model, image_features, inputs_embeds, input_ids, attention_mask, labels):
    num_images, num_image_patches, embed_dim = image_features.shape
    batch_size, sequence_length = input_ids.shape
    left_padding = not torch.sum(input_ids[:, -1] == torch.tensor(model.pad_token_id))
    special_image_token_mask = input_ids == model.config.image_token_index
    num_special_image_tokens = torch.sum(special_image_token_mask, dim=-1)
    max_embed_dim = (num_special_image_tokens.max() * (num_image_patches - 1)) + sequence_length
    batch_indices, non_image_indices = torch.where(input_ids != model.config.image_token_index)
    new_token_positions = torch.cumsum((special_image_token_mask * (num_image_patches - 1) + 1), -1) - 1
    nb_image_pad = max_embed_dim - 1 - new_token_positions[:, -1]
    if left_padding:
        new_token_positions += nb_image_pad[:, None]  # offset for left padding
    text_to_overwrite = new_token_positions[batch_indices, non_image_indices]
    final_embedding = torch.zeros(
        batch_size, max_embed_dim, embed_dim, dtype=inputs_embeds.dtype, device=inputs_embeds.device)
    final_attention_mask = torch.zeros(
        batch_size, max_embed_dim, dtype=attention_mask.dtype, device=inputs_embeds.device)
    target_device = inputs_embeds.device
    batch_indices, non_image_indices, text_to_overwrite = (
        batch_indices.to(target_device),
        non_image_indices.to(target_device),
        text_to_overwrite.to(target_device),)
    attention_mask = attention_mask.to(target_device)
    final_embedding[batch_indices, text_to_overwrite] = inputs_embeds[batch_indices, non_image_indices]
    final_attention_mask[batch_indices, text_to_overwrite] = attention_mask[batch_indices, non_image_indices]
    image_to_overwrite = torch.all(final_embedding == 0, dim=-1)
    image_to_overwrite &= image_to_overwrite.cumsum(-1) - 1 >= nb_image_pad[:, None].to(target_device)
    if image_to_overwrite.sum() != image_features.shape[:-1].numel():
        raise ValueError(
            f"The input provided to the model are wrong. The number of image tokens is {torch.sum(special_image_token_mask)} while"
            f" the number of image given to the model is {num_images}. This prevents correct indexing and breaks batch generation."
        )
    final_embedding[image_to_overwrite] = image_features.contiguous().reshape(-1, embed_dim).to(target_device)
    final_attention_mask |= image_to_overwrite
    position_ids = (final_attention_mask.cumsum(-1) - 1).masked_fill_((final_attention_mask == 0), 1)
    final_labels = None
    if labels is None:
        final_labels = None

    return final_embedding, final_attention_mask, final_labels, position_ids


def logits2tokens(logits, k):
    probs = torch.softmax(logits, dim=-1)
    top_k_probs, top_k_ids = torch.topk(probs, k, dim=-1)
    token_ids = top_k_ids.cpu().numpy()
    token_sequences = []

    for batch in range(token_ids.shape[0]):
        token_sequence = []
        for token_id in token_ids[0]:
            seq = []
            for tid in token_id:
                tokens = tokenizer.decode(tid, skip_special_tokens=True)
                seq.append(tokens)
            token_sequence.append(seq)
        token_sequences.append(token_sequence)

    return token_sequences
