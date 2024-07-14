from PIL import Image
import inspect
import torch
from encoder import LlavaEncoder
from projector import LlavaProjector
from decoder import LlavaDecoder
from init import model, processor, pad_token_id, stopping_criteria, \
    generation_config, logits_processor, max_length
from utils import merge_input_ids_with_image_features, Config, logits2tokens


def init_generate(image_file, prompt, eos_token_id, model_kwargs):
    raw_image = Image.open(image_file)
    generation_config.do_sample = False
    inputs = processor(prompt, raw_image, return_tensors='pt').to(0, torch.float16)

    input_ids = inputs['input_ids']
    pixel_values = inputs['pixel_values']
    attention_mask = inputs['attention_mask']

    # init variables
    inputs_tensor, model_input_name, model_kwargs = model._prepare_model_inputs(
        input_ids, generation_config.bos_token_id, model_kwargs)
    batch_size = inputs_tensor.shape[0]
    model_kwargs["output_attentions"] = generation_config.output_attentions
    model_kwargs["output_hidden_states"] = generation_config.output_hidden_states
    if not model.config.is_encoder_decoder and model_input_name == "inputs_embeds":
        model_kwargs["use_cache"] = True
    else:
        model_kwargs["use_cache"] = generation_config.use_cache
    accepts_attention_mask = "attention_mask" in set(inspect.signature(model.forward).parameters.keys())
    requires_attention_mask = "encoder_outputs" not in model_kwargs
    if model_kwargs.get("attention_mask", None) is None and requires_attention_mask and accepts_attention_mask:
        model_kwargs["attention_mask"] = model._prepare_attention_mask_for_generation(
            inputs_tensor, generation_config.pad_token_id, generation_config.eos_token_id)
    if model.config.is_encoder_decoder and "encoder_outputs" not in model_kwargs:
        model_kwargs = model._prepare_encoder_decoder_kwargs_for_generation(
            inputs_tensor, model_kwargs, model_input_name)
    if model.config.is_encoder_decoder:
        input_ids, model_kwargs = model._prepare_decoder_input_ids_for_generation(
            batch_size=batch_size,
            model_input_name=model_input_name,
            model_kwargs=model_kwargs,
            decoder_start_token_id=generation_config.decoder_start_token_id,
            bos_token_id=generation_config.bos_token_id,
            device=inputs_tensor.device, )
    else:
        input_ids = inputs_tensor if model_input_name == "input_ids" else model_kwargs.pop("input_ids")
    input_ids_length = input_ids.shape[-1]
    has_default_max_length = max_length is None and generation_config.max_length is not None
    if generation_config.max_new_tokens is not None:
        generation_config.max_length = generation_config.max_new_tokens + input_ids_length
    model._validate_generated_length(generation_config, input_ids_length, has_default_max_length)
    if isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]
    eos_token_id_tensor = torch.tensor(eos_token_id).to(input_ids.device) if eos_token_id is not None else None
    output_scores = model.generation_config.output_scores
    output_attentions = model.generation_config.output_attentions
    output_hidden_states = model.generation_config.output_hidden_states = True
    return_dict_in_generate = model.generation_config.return_dict_in_generate
    scores = () if (return_dict_in_generate and output_scores) else None

    return input_ids, pixel_values, attention_mask, input_ids_length, inputs_tensor, eos_token_id_tensor, scores


def internal_contrast_generate(image_file, prompt, alpha,
                               contrast_layer_id, eos_token_id,
                               labels, mode, model_kwargs):
    encoder = LlavaEncoder()
    projector = LlavaProjector(model)
    decoder = LlavaDecoder()
    encoder.training = False
    projector.training = False
    decoder.training = False
    vision_feature_select_strategy = 'default'

    input_ids, pixel_values, attention_mask, input_ids_length, \
    inputs_tensor, eos_token_id_tensor, scores = init_generate(image_file, prompt, eos_token_id, model_kwargs)

    unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)

    prepared_logits_processor = model._get_logits_processor(
        prefix_allowed_tokens_fn=None,
        generation_config=generation_config,
        input_ids_seq_length=input_ids_length,
        encoder_input_ids=inputs_tensor,
        logits_processor=logits_processor,
        model_kwargs=model_kwargs, )
    prepared_stopping_criteria = model._get_stopping_criteria(
        generation_config=generation_config, stopping_criteria=stopping_criteria)

    # loop to generate text
    this_peer_finished = False
    is_get_token_size = False
    token_size = 0
    while True:
        inputs_embeds = decoder.get_input_embeddings()(input_ids)
        image_outputs = encoder(pixel_values)
        if vision_feature_select_strategy == "default":
            image_outputs = image_outputs[:, 1:]
        elif vision_feature_select_strategy == "full":
            image_outputs = image_outputs
        image_features = projector(image_outputs)
        inputs_embeds, attention_mask, labels, position_ids = merge_input_ids_with_image_features(
            model, image_features, inputs_embeds, input_ids, attention_mask, labels)

        config = Config()
        outputs = decoder(
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_hidden_states=config.output_hidden_states,
            past_key_values=config.past_key_values,
            use_cache=config.use_cache,
            output_attentions=config.output_attentions,
            return_dict=config.return_dict,
        )

        last_logits = decoder.model.lm_head(outputs.hidden_states[-1])
        contrast_logits = decoder.model.lm_head(outputs.hidden_states[contrast_layer_id])
        if mode == 1:
            logits = (1 - alpha) * last_logits + alpha * contrast_logits
        elif mode == 2:
            logits = (1 + alpha) * contrast_logits - alpha * last_logits
        else:
            logits = decoder.model.lm_head(outputs.hidden_states[-1])

        if not is_get_token_size:
            token_size = logits.size(1)
            is_get_token_size = True
        next_token_logits = logits[:, -1, :]
        next_tokens_scores = prepared_logits_processor(input_ids, next_token_logits)
        # if return_dict_in_generate:
        #     if output_scores:
        #         scores += (next_tokens_scores,)
        #     if output_attentions:
        #         decoder_attentions += (
        #             (outputs.decoder_attentions,) if model.config.is_encoder_decoder else (outputs.attentions,))
        #         if model.config.is_encoder_decoder:
        #             cross_attentions += (outputs.cross_attentions,)
        #     if output_hidden_states:
        #         outputs.hidden_states += (
        #             (outputs.decoder_hidden_states,)
        #             if model.config.is_encoder_decoder
        #             else (outputs.hidden_states,))

        next_tokens = torch.argmax(next_tokens_scores, dim=-1)

        if eos_token_id is not None:
            if pad_token_id is None:
                raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        model_kwargs = model._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder=model.config.is_encoder_decoder)
        if eos_token_id_tensor is not None:
            unfinished_sequences = unfinished_sequences.mul(
                next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0))
            if unfinished_sequences.max() == 0:
                this_peer_finished = True
        if prepared_stopping_criteria(input_ids, scores):
            this_peer_finished = True
        if this_peer_finished:
            break

    print(processor.decode(input_ids[0], skip_special_tokens=True))
    print("--------------------------------")
    print("last layer:\n", logits2tokens(logits[:, token_size - 1:, :], 10))
    print("--------------------------------")

    return logits, token_size


def external_contrast_generate(image_file1, prompt, image_file2, alpha,
                               eos_token_id, labels, mode, model_kwargs):
    encoder = LlavaEncoder()
    projector = LlavaProjector(model)
    decoder = LlavaDecoder()
    encoder.training = False
    projector.training = False
    decoder.training = False
    vision_feature_select_strategy = 'default'

    input_ids, pixel_values1, attention_mask1, input_ids_length, \
    inputs_tensor, eos_token_id_tensor, scores = init_generate(image_file1, prompt, eos_token_id, model_kwargs)

    input_ids, pixel_values2, attention_mask2, input_ids_length, \
    inputs_tensor, eos_token_id_tensor, scores = init_generate(image_file2, prompt, eos_token_id, model_kwargs)

    unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)

    prepared_logits_processor = model._get_logits_processor(
        prefix_allowed_tokens_fn=None,
        generation_config=generation_config,
        input_ids_seq_length=input_ids_length,
        encoder_input_ids=inputs_tensor,
        logits_processor=logits_processor,
        model_kwargs=model_kwargs, )
    prepared_stopping_criteria = model._get_stopping_criteria(
        generation_config=generation_config, stopping_criteria=stopping_criteria)

    # loop to generate text
    this_peer_finished = False
    is_get_token_size = False
    token_size = 0
    while True:
        torch.cuda.empty_cache()
        inputs_embeds = decoder.get_input_embeddings()(input_ids)
        image_outputs1 = encoder(pixel_values1)
        image_outputs2 = encoder(pixel_values2)
        if vision_feature_select_strategy == "default":
            image_outputs1 = image_outputs1[:, 1:]
        elif vision_feature_select_strategy == "full":
            image_outputs1 = image_outputs1
        if vision_feature_select_strategy == "default":
            image_outputs2 = image_outputs2[:, 1:]
        elif vision_feature_select_strategy == "full":
            image_outputs2 = image_outputs2
        image_features1 = projector(image_outputs1)
        image_features2 = projector(image_outputs2)

        inputs_embeds1, attention_mask1, labels1, position_ids1 = merge_input_ids_with_image_features(
            model, image_features1, inputs_embeds, input_ids, attention_mask1, labels)
        inputs_embeds2, attention_mask2, labels2, position_ids2 = merge_input_ids_with_image_features(
            model, image_features2, inputs_embeds, input_ids, attention_mask2, labels)

        config = Config()
        outputs1 = decoder(
            attention_mask=attention_mask1,
            position_ids=position_ids1,
            inputs_embeds=inputs_embeds1,
            output_hidden_states=config.output_hidden_states,
            past_key_values=config.past_key_values,
            use_cache=config.use_cache,
            output_attentions=config.output_attentions,
            return_dict=config.return_dict,
        )
        outputs2 = decoder(
            attention_mask=attention_mask2,
            position_ids=position_ids2,
            inputs_embeds=inputs_embeds2,
            output_hidden_states=config.output_hidden_states,
            past_key_values=config.past_key_values,
            use_cache=config.use_cache,
            output_attentions=config.output_attentions,
            return_dict=config.return_dict,
        )

        logits1 = decoder.model.lm_head(outputs1.hidden_states[-1])
        logits2 = decoder.model.lm_head(outputs2.hidden_states[-1])

        if mode == 1:
            logits = (1 - alpha) * logits1 + alpha * logits2
        elif mode == 2:
            logits = (1 + alpha) * logits1 - alpha * logits2
        else:
            logits = decoder.model.lm_head(outputs1.hidden_states[-1])

        if not is_get_token_size:
            token_size = logits.size(1)
            is_get_token_size = True
        next_token_logits = logits[:, -1, :]
        next_tokens_scores = prepared_logits_processor(input_ids, next_token_logits)
        # if return_dict_in_generate:
        #     if output_scores:
        #         scores += (next_tokens_scores,)
        #     if output_attentions:
        #         decoder_attentions += (
        #             (outputs.decoder_attentions,) if model.config.is_encoder_decoder else (outputs.attentions,))
        #         if model.config.is_encoder_decoder:
        #             cross_attentions += (outputs.cross_attentions,)
        #     if output_hidden_states:
        #         outputs.hidden_states += (
        #             (outputs.decoder_hidden_states,)
        #             if model.config.is_encoder_decoder
        #             else (outputs.hidden_states,))

        next_tokens = torch.argmax(next_tokens_scores, dim=-1)

        if eos_token_id is not None:
            if pad_token_id is None:
                raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        model_kwargs = model._update_model_kwargs_for_generation(
            outputs1, model_kwargs, is_encoder_decoder=model.config.is_encoder_decoder)
        if eos_token_id_tensor is not None:
            unfinished_sequences = unfinished_sequences.mul(
                next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0))
            if unfinished_sequences.max() == 0:
                this_peer_finished = True
        if prepared_stopping_criteria(input_ids, scores):
            this_peer_finished = True
        if this_peer_finished:
            break
        del image_outputs1, image_outputs2, image_features1, \
            image_features2, outputs1, outputs2, logits1, logits2

    print(processor.decode(input_ids[0], skip_special_tokens=True))
    print("--------------------------------")
    print("last layer:\n", logits2tokens(logits[:, token_size - 1:, :], 10))
    print("--------------------------------")

    return logits, token_size
