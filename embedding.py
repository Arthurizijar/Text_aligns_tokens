import torch
from llm2vec import LLM2Vec
import torch.nn.functional as F

def get_embeddings(model_name, model_list, data_name, texts, isdoc=False, islayer=False):

    model, tokenizer, doc_model, doc_tokenizer = model_list[0], model_list[1], model_list[2], model_list[3]
    if model_name == "bert":
        inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            embs = model(**inputs, output_hidden_states=True, return_dict=True).pooler_output
    elif model_name in ["prompt_bert", "prompt_bert_sup", "prompt_bert_unsup"]:
        template = "This_sentence_:_\"*sent_0*\"_means_*mask*."
        template = template.replace('*mask*', tokenizer.mask_token).replace('_', ' ')
        texts = [template.replace('*sent 0*', s).strip() for s in texts]
        inputs = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True, return_dict=True)
            last_hidden = outputs.last_hidden_state
            mask_matrix = (inputs['input_ids'] == tokenizer.mask_token_id)
            # 找到每行最后一个True的索引
            last_true_indices = torch.tensor(
                [row.nonzero(as_tuple=True)[0][-1] if row.any() else 0 for row in mask_matrix])
            # 将除了每行最后一个True之外的True置为False
            for i in range(mask_matrix.size(0)):
                if last_true_indices[i] > 0:
                    mask_matrix[i, :last_true_indices[i]] = False
            pooler_output = last_hidden[mask_matrix]
            embs = pooler_output.view(inputs['input_ids'].shape[0], -1)
    elif model_name == "simcse":
        inputs = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            embs = model(**inputs, output_hidden_states=True, return_dict=True).pooler_output
    elif model_name == "sentence_t5":
        embs = model.encode(texts, convert_to_tensor=True)
    elif model_name == "contriever":
        inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            embs = model(**inputs)
            embs = embs[0].masked_fill(~inputs['attention_mask'][..., None].bool(), 0.)
            embs = embs.sum(dim=1) / inputs['attention_mask'].sum(dim=1)[..., None]
        
    elif model_name in ["dpr", "spider"]:
        if isdoc and model_name == "dpr":
            inputs = doc_tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            with torch.no_grad():
                embs = doc_model(**inputs).pooler_output
        else:
            inputs = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            with torch.no_grad():
                embs = model(**inputs).pooler_output
    elif model_name in ["gpt_neo", "sgpt_nli", "sgpt_msmarco"]:
        if islayer:
            with torch.no_grad():
                texts = {"input_ids": torch.tensor(texts).to(model.device)}
                embs = model(**texts, output_hidden_states=True, return_dict=True).last_hidden_state[:, 0, :]
            return embs
        if isdoc:
            batch_size = 16
        else:
            batch_size = 128
        embs = None
        for ii in range(0, len(texts), batch_size):
            # print(ii)
            batch_texts = texts[ii: ii + batch_size]
            inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=512, return_tensors='pt')
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            with torch.no_grad():
                last_hidden_state = model(**inputs, output_hidden_states=True, return_dict=True).hidden_states[-1]
                weights = (
                    torch.arange(start=1, end=last_hidden_state.shape[1] + 1)
                    .unsqueeze(0)
                    .unsqueeze(-1)
                    .expand(last_hidden_state.size())
                    .float().to(last_hidden_state.device)
                )
                # Get attn mask of shape [bs, seq_len, hid_dim]
                input_mask_expanded = (
                    inputs["attention_mask"]
                    .unsqueeze(-1)
                    .expand(last_hidden_state.size())
                    .float()
                )
                # Perform weighted mean pooling across seq_len: bs, seq_len, hidden_dim -> bs, hidden_dim
                sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded * weights, dim=1)
                sum_mask = torch.sum(input_mask_expanded * weights, dim=1)
                batch_embs = (sum_embeddings / sum_mask).cpu()
                if embs is None:
                    embs = batch_embs
                else:
                    embs = torch.cat((embs, batch_embs), dim=0)

    elif model_name in ["opt", "opt_eol", "opt_eol_icl", "opt_eol_cse", "opt_eol_cse_ours"]:
        if model_name in ["opt_eol", "opt_eol_cse", "opt_eol_cse_ours"]:
            template = 'This_sentence_:_"*sent_0*"_means_in_one_word:"'
        elif model_name in ["opt"]:
            template = '*sent_0*'
        elif model_name in ["opt_eol_icl"]:
            template = ('This_sentence_:_"relating_to_switzerland_or_its_people."_means_in_one_word:"Swiss'
                        '".This_sentence_:_"*sent_0*"_means_in_one_word:"')
        else:
            assert NotImplementedError
        if isdoc:
            batch_size = 16
        else:
            batch_size = 64
        embs = None
        for ii in range(0, len(texts), batch_size):
            # print(ii)
            batch_texts = texts[ii: ii + batch_size]
            inputs = tokenizer([template.replace('*sent_0*', i).replace('_', ' ') for i in batch_texts],
                               padding=True, return_tensors="pt")
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            with torch.no_grad():
                batch_embs = model(**inputs, output_hidden_states=True, return_dict=True).hidden_states[-1][:, -1, :].cpu()
            if embs is None:
                embs = batch_embs
            else:
                embs = torch.cat((embs, batch_embs), dim=0)
    elif model_name in ["llama", "llama_eol", "llama_eol_cse", "llama2", "llama2_eol"]:
        if model_name in ["llama", "llama2"]:
            template = "*sent_0*"
        elif model_name in ["llama_eol", "llama_eol_cse"]:
            template = 'This_sentence_:_"*sent_0*"_means_in_one_word:"'
        else:
            raise NotImplementedError
        if isdoc:
            batch_size = 4
        else:
            batch_size = 64
        embs = None
        for ii in range(0, len(texts), batch_size):
            batch_texts = texts[ii: ii + batch_size]
            inputs = tokenizer([template.replace('*sent_0*', i).replace('_', ' ') for i in batch_texts], padding=True,
                               max_length=512, return_tensors="pt")
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            with torch.no_grad():
                batch_embs = model(**inputs, output_hidden_states=True,
                                   return_dict=True).hidden_states[-1][:, -1, :].cpu()
            if embs is None:
                embs = batch_embs
            else:
                embs = torch.cat((embs, batch_embs), dim=0)
    elif model_name == "gte":
        embs = None
        if isdoc:
            batch_size = 8
        else:
            batch_size = 64
        for ii in range(0, len(texts), batch_size):
            batch_texts = texts[ii: ii + batch_size]
            inputs = tokenizer(batch_texts, max_length=512, padding=True, truncation=True, return_tensors='pt')
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs)
            last_hidden_states = outputs.last_hidden_state
            last_hidden = last_hidden_states.masked_fill(~inputs['attention_mask'][..., None].bool(), 0.0)
            batch_embs = last_hidden.sum(dim=1) / inputs['attention_mask'].sum(dim=1)[..., None]
            batch_embs = F.normalize(batch_embs, p=2, dim=1)
            if embs is None:
                embs = batch_embs
            else:
                embs = torch.cat((embs, batch_embs), dim=0)
    elif model_name == "e5":
        if not isdoc:
            texts = ["query: " + s for s in texts]
        else:
            texts = ["passage: " + s for s in texts]
        embs = None
        if isdoc:
            batch_size = 8
        else:
            batch_size = 64
        for ii in range(0, len(texts), batch_size):
            batch_texts = texts[ii: ii + batch_size]
            inputs = tokenizer(batch_texts, max_length=512, padding=True, truncation=True, return_tensors='pt')
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs)
                last_hidden_states = outputs.last_hidden_state
                last_hidden = last_hidden_states.masked_fill(~inputs['attention_mask'][..., None].bool(), 0.0)
                batch_embs = last_hidden.sum(dim=1) / inputs['attention_mask'].sum(dim=1)[..., None]
                batch_embs = F.normalize(batch_embs, p=2, dim=1)
                if embs is None:
                    embs = batch_embs
                else:
                    embs = torch.cat((embs, batch_embs), dim=0)
    elif model_name == "e5_mistral":
        def last_token_pool(last_hidden_states, attention_mask):
            left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
            if left_padding:
                return last_hidden_states[:, -1]
            else:
                sequence_lengths = attention_mask.sum(dim=1) - 1
                batch_size = last_hidden_states.shape[0]
                return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

        def get_detailed_instruct(task_description: str, query: str) -> str:
            return f'Instruct: {task_description}\nQuery: {query}'

        if data_name == "sts":
            task = 'Given a web search query, retrieve relevant passages that answer the query'
            texts = [get_detailed_instruct(task, t) for t in texts]
        # Tokenize the input texts
        embs = None
        if isdoc:
            batch_size = 4
        else:
            batch_size = 64
        for ii in range(0, len(texts), batch_size):
            batch_texts = texts[ii: ii + batch_size]
            max_length = 4096
            batch_dict = tokenizer(batch_texts, max_length=max_length - 1, return_attention_mask=False, padding=False,
                                truncation=True)
            # append eos_token_id to every input_ids
            batch_dict['input_ids'] = [input_ids + [tokenizer.eos_token_id] for input_ids in batch_dict['input_ids']]
            batch_dict = tokenizer.pad(batch_dict, padding=True, return_attention_mask=True, return_tensors='pt')
            batch_dict = {k: v.to(model.device) for k, v in batch_dict.items()}
            with torch.no_grad():
                outputs = model(**batch_dict)
                batch_embs = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
                batch_embs = F.normalize(batch_embs, p=2, dim=1)
            if embs is None:
                embs = batch_embs
            else:
                embs = torch.cat((embs, batch_embs), dim=0)
    elif model_name == "syncse":
        inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            embs = model(**inputs, output_hidden_states=True, return_dict=True).pooler_output
    elif model_name in ["llm2vec_mistral_unsup", "llm2vec_mistral_sup"]:
        l2v = LLM2Vec(model, tokenizer, pooling_mode="mean", max_length=512)
        if not isdoc:
            instruction = "Given a web search query, retrieve relevant passages that answer the query:"
            texts = [[instruction, t] for t in texts]
        embs = l2v.encode(texts).cuda()
    elif model_name in ["mistral", "gritlm"]:
        instruction = "Given a scientific paper title, retrieve the paper's abstract"
        def gritlm_instruction(instruction):
            return "<|user|>\n" + instruction + "\n<|embed|>\n" if instruction else "<|embed|>\n"

        if isdoc:
            embs = model.encode(texts, instruction=gritlm_instruction(""), batch_size=128)
            embs = torch.from_numpy(embs)
        else:
            embs = model.encode(texts, instruction=gritlm_instruction(instruction), batch_size=128)
            embs = torch.from_numpy(embs)
    elif model_name == "gte_qwen2":

        def last_token_pool(last_hidden_states, attention_mask):
            left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
            if left_padding:
                return last_hidden_states[:, -1]
            else:
                sequence_lengths = attention_mask.sum(dim=1) - 1
                batch_size = last_hidden_states.shape[0]
                return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

        def get_detailed_instruct(task_description: str, query: str) -> str:
            return f'Instruct: {task_description}\nQuery: {query}'

        task = 'Given a web search query, retrieve relevant passages that answer the query'
        max_length = 8192
        if not isdoc:
            texts = [get_detailed_instruct(task, sent) for sent in texts]
        # Tokenize the input texts
        batch_dict = tokenizer(texts, max_length=max_length, padding=True, truncation=True, return_tensors='pt')
        outputs = model(**batch_dict, output_hidden_states=True)
        embs = last_token_pool(outputs.hidden_states[-1], batch_dict['attention_mask'])
    elif model_name == "mpnet":
        def mean_pooling(model_output, attention_mask):
            token_embeddings = model_output[0]
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

        encoded_input = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            model_output = model(**encoded_input)
        sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
        embs = F.normalize(sentence_embeddings, p=2, dim=1)
    else:
        raise NotImplementedError
    return embs.to(torch.float32)


def get_logits(args, decoder_layer, embs):
    model_name = args.model
    if model_name != "sentence_t5":
        with torch.no_grad():
            logits = decoder_layer(embs).cpu()
    else:
        assert len(decoder_layer) == 2
        decoder, lm_head = decoder_layer[0], decoder_layer[1]
        # 0 is pad_id in T5
        decoder_input_ids = torch.tensor([[0]] * len(embs), device=decoder.device)
        with torch.no_grad():
            decoder_outputs = decoder(
                encoder_hidden_states=embs.unsqueeze(1),
                input_ids=decoder_input_ids,
            )
            sequence_output = decoder_outputs[0]
            logits = lm_head(sequence_output).squeeze(1).cpu()
    return logits