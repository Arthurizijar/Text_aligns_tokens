from transformers import AutoModel, AutoTokenizer, DPRContextEncoder, DPRContextEncoderTokenizer, AutoConfig, AutoModelForCausalLM, AutoModelForMaskedLM, T5ForConditionalGeneration
from gritlm import GritLM
from peft import PeftModel
from functools import partial

import torch
import os


mlm_head = None
model = None
tokenizer = None
doc_model = None
doc_tokenizer = None

def load_model(args, model_name):
    model, tokenizer, doc_model, doc_tokenizer = None, None, None, None
    if model_name == "bert":
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        model = AutoModel.from_pretrained("bert-base-uncased").cuda().eval()
    elif model_name == "prompt_bert":
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        model = AutoModel.from_pretrained("bert-base-uncased").cuda().eval()
    elif model_name == "prompt_bert_sup":
        tokenizer = AutoTokenizer.from_pretrained("royokong/sup-PromptBERT")
        model = AutoModel.from_pretrained("royokong/sup-PromptBERT").cuda().eval()
    elif model_name == "prompt_bert_unsup":
        tokenizer = AutoTokenizer.from_pretrained("royokong/unsup-PromptBERT")
        model = AutoModel.from_pretrained("royokong/unsup-PromptBERT").cuda().eval()
    elif model_name == "simcse":
        tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased")
        model = AutoModel.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased").cuda().eval()
    elif model_name == "sentence_t5":
        from sentence_transformers import SentenceTransformer
        tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-base")
        model = SentenceTransformer('sentence-transformers/sentence-t5-base').cuda().eval()
    elif model_name == "contriever":
        # from src.contriever import Contriever
        tokenizer = AutoTokenizer.from_pretrained("facebook/contriever")
        model = AutoModel.from_pretrained("facebook/contriever").cuda().eval()
    elif model_name == "dpr":
        doc_tokenizer = DPRContextEncoderTokenizer.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base')
        doc_model = DPRContextEncoder.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base').cuda().eval()
        tokenizer = AutoTokenizer.from_pretrained('facebook/dpr-question_encoder-single-nq-base')
        model = AutoModel.from_pretrained('facebook/dpr-question_encoder-single-nq-base').cuda().eval()
    elif model_name == "gpt_neo":
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")
        tokenizer.add_special_tokens(
            {"bos_token": "<|endoftext|>", "eos_token": "<|endoftext|>", "unk_token": "<|endoftext|>",
             "pad_token": "<|endoftext|>"})
        model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B").cuda().eval()
    elif model_name == "sgpt_nli":
        tokenizer = AutoTokenizer.from_pretrained("Muennighoff/sgpt-1.3B-weightedmean-nli")
        model = AutoModel.from_pretrained("Muennighoff/sgpt-1.3B-weightedmean-nli").cuda().eval()
    elif model_name == "sgpt_msmarco":
        tokenizer = AutoTokenizer.from_pretrained("Muennighoff/sgpt-1.3B-weightedmean-msmarco-specb-bitfit")
        model = AutoModel.from_pretrained("Muennighoff/sgpt-1.3B-weightedmean-msmarco-specb-bitfit").cuda().eval()
    elif model_name in ["opt", "opt_eol", "opt_eol_icl"]:
        tokenizer = AutoTokenizer.from_pretrained("facebook/opt-1.3b")
        model = AutoModelForCausalLM.from_pretrained("facebook/opt-1.3b").cuda().eval()
        tokenizer.pad_token_id = 0
        tokenizer.padding_side = "left"
    elif model_name in ["llama", "llama_eol"]:
        tokenizer = AutoTokenizer.from_pretrained(args.llama_path)
        model = AutoModelForCausalLM.from_pretrained(args.llama_path,
                                                     torch_dtype=torch.bfloat16).cuda().eval()
        tokenizer.pad_token_id = 0
        tokenizer.padding_side = "left"
    elif model_name in ["llama2", "llama2_eol"]:
        tokenizer = AutoTokenizer.from_pretrained(args.llama_path)
        model = AutoModelForCausalLM.from_pretrained(args.llama_path,
                                                     torch_dtype=torch.bfloat16,
                                                     device_map="cuda" if torch.cuda.is_available() else "cpu",).eval()
        tokenizer.pad_token_id = 0
        tokenizer.padding_side = "left"
    elif model_name == "gte":
        tokenizer = AutoTokenizer.from_pretrained("thenlper/gte-base")
        model = AutoModel.from_pretrained("thenlper/gte-base").cuda().eval()
    elif model_name == "e5":
        tokenizer = AutoTokenizer.from_pretrained('intfloat/e5-base-v2')
        model = AutoModel.from_pretrained('intfloat/e5-base-v2').cuda().eval()
    elif model_name == "llama_eol_cse":
        tokenizer = AutoTokenizer.from_pretrained(args.llama_path)
        ori_model = AutoModelForCausalLM.from_pretrained(args.llama_path, torch_dtype=torch.bfloat16).cuda().eval()
        tokenizer.pad_token_id = 0
        tokenizer.padding_side = "left"
        model = PeftModel.from_pretrained(ori_model, "royokong/prompteol-llama-7b", torch_dtype=torch.float16)
    elif model_name == "opt_eol_cse":
        tokenizer = AutoTokenizer.from_pretrained("facebook/opt-1.3b")
        ori_model = AutoModelForCausalLM.from_pretrained("facebook/opt-1.3b").cuda().eval()
        tokenizer.pad_token_id = 0
        tokenizer.padding_side = "left"
        model = PeftModel.from_pretrained(ori_model, "royokong/prompteol-opt-1.3b", torch_dtype=torch.float16)
    elif model_name == "mistral":
        model = GritLM("mistralai/Mistral-7B-Instruct-v0.2", torch_dtype="auto", attn=None)
        tokenizer = model.tokenizer
    elif model_name == "e5_mistral":
        tokenizer = AutoTokenizer.from_pretrained('intfloat/e5-mistral-7b-instruct')
        model = AutoModel.from_pretrained('intfloat/e5-mistral-7b-instruct').cuda().eval()
    elif model_name in ["llm2vec_mistral_unsup", "llm2vec_mistral_sup"]:
        tokenizer = AutoTokenizer.from_pretrained("McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp")
        config = AutoConfig.from_pretrained("McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp", trust_remote_code=True)
        model = AutoModel.from_pretrained(
            "McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp",
            trust_remote_code=True,
            config=config,
            torch_dtype=torch.bfloat16,
            device_map="cuda" if torch.cuda.is_available() else "cpu",
        )
        model = PeftModel.from_pretrained(model, "McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp")
        model = model.merge_and_unload()
        if model_name == "llm2vec_mistral_unsup":
            model = PeftModel.from_pretrained(model, "McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp-unsup-simcse")
        elif model_name == "llm2vec_mistral_sup":
            model = PeftModel.from_pretrained(model, "McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp-supervised")
    elif model_name == "gritlm":
        model = GritLM("GritLM/GritLM-7B", torch_dtype="auto")
        tokenizer = model.tokenizer
    elif model_name == "gte_qwen2":
        model = AutoModelForCausalLM.from_pretrained('Alibaba-NLP/gte-Qwen2-7B-instruct', trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained('Alibaba-NLP/gte-Qwen2-7B-instruct', trust_remote_code=True)
    elif model_name == "mpnet":
        model = AutoModel.from_pretrained('sentence-transformers/all-mpnet-base-v2')
        tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')
    elif model_name == "spider":
        model = DPRContextEncoder.from_pretrained("tau/spider")
        tokenizer = AutoTokenizer.from_pretrained("tau/spider")
    else:
        raise NotImplementedError
    return model, tokenizer, doc_model, doc_tokenizer




# def get_embedding_weights(model_name, model):
#     if model_name == "bert":
#         return model.embeddings.word_embeddings.weight.data
#     elif model_name == "prompt_bert":
#         return model.bert.embeddings.word_embeddings.weight.data
#     elif model_name in ["prompt_bert_sup", "prompt_bert_unsup"]:
#         return model.embeddings.word_embeddings.weight.data
#     elif model_name == "simcse":
#         return model.embeddings.word_embeddings.weight.data
#     elif model_name == "sentence_t5":
#         # print(model._modules.keys())
#         return model._modules['0'].auto_model.shared.weight.data
#     elif model_name == "contriever":
#         return model.embeddings.word_embeddings.weight.data
#     elif model_name == "dpr":
#         return model.question_encoder.bert_model.embeddings.word_embeddings.weight.data
#     elif model_name in ["gpt_neo", "sgpt_nli", "sgpt_msmarco"]:
#         return model.wte.weight
#     elif model_name in ["opt_eol", "opt_eol_icl"]:
#         return model.model.decoder.get_input_embeddings().weight.data
#     elif model_name == "gte":
#         return model.embeddings.word_embeddings.weight.data
#     elif model_name == "e5":
#         return model.embeddings.word_embeddings.weight.data
#     elif model_name == "llama_eol_cse":
#         return model.model.model.embed_tokens.weight.data
#     elif model_name == "opt_eol_cse":
#         return model.model.model.decoder.get_input_embeddings().weight.data
#     elif model_name == "opt_eol_cse_ours":
#         return model.model.model.decoder.get_input_embeddings().weight.data
#     elif model_name == "syncse":
#         return model.embeddings.word_embeddings.weight.data
#     elif model_name == "mistral":
#         return model.embed_tokens.weight.data.float()
#     elif model_name in ["llm2vec_mistral", "llm2vec_mistral_sup"]:
#         return model.model.embed_tokens.weight.data.float()
#     elif model_name in ["mistral", "gritlm"]:
#         return model.model.model.embed_tokens.weight.data.float()
#     else:
#         raise NotImplementedError


def load_decoder_layer(args, model=None):
    model_name = args.model
    if model_name in ["bert", "prompt_bert", "prompt_bert_sup", "prompt_bert_unsup", "simcse", "contriever", "dpr", "gte", "e5", "spider"]:
        return AutoModelForMaskedLM.from_pretrained("bert-base-uncased").cuda().eval().cls
    elif model_name == "mpnet":
        return AutoModelForMaskedLM.from_pretrained("microsoft/mpnet-base").cuda().eval().lm_head
    elif model_name == "sentence_t5":
        t5 = T5ForConditionalGeneration.from_pretrained("google-t5/t5-base").cuda().eval()
        return (t5.decoder, t5.lm_head)
    elif model_name in ["gpt_neo", "opt", "opt_eol", "llama", "llama_eol", "llama2", "llama2_eol", "gte_qwen2"]:
        return model.lm_head
    elif model_name in ["opt_eol_cse", "llama_eol_cse", "mistral", "gritlm"]:
        return model.model.lm_head
    elif model_name in ["sgpt_nli", "sgpt_msmarco", "llm2vec_mistral", "llm2vec_mistral_sup", "llm2vec_llama2", "llm2vec_llama2_sup", "e5_mistral"]:
        weights_path = args.weight_dir + f"{model_name}_linear.pth"
        if os.path.exists(weights_path):
            import torch.nn as nn
            state_dict = torch.load(weights_path)
            shape = state_dict['weight'].shape
            linear_layer = nn.Linear(shape[1], shape[0], bias=False).cuda()
            linear_layer.load_state_dict(state_dict)
            return linear_layer
        else:
            if model_name in ["e5_mistral"]:
                temp_model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1", low_cpu_mem_usage=True).cuda().eval()
            elif model_name in ["llm2vec_mistral", "llm2vec_mistral_sup"]:
                temp_model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", low_cpu_mem_usage=True).cuda().eval()
            elif model_name in ["llm2vec_llama2", "llm2vec_llama2_sup"]:
                temp_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", low_cpu_mem_usage=True).cuda().eval()
            elif model_name in ["sgpt_nli", "sgpt_msmarco"]:
                temp_model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B").cuda().eval()
            else:
                raise NotImplementedError
            torch.save(temp_model.lm_head.state_dict(), weights_path)
    else:
        raise NotImplementedError


# def load_mlm_head(model_name):
#     assert model_name in ["bert", "simcse", "contriever", "gte", "e5", "dpr", "syncse", "prompt_bert", "mpnet", "spider"]
#     if model_name == "syncse":
#         mlm_head = AutoModelForMaskedLM.from_pretrained("roberta-large").cuda().eval().lm_head
#     elif model_name == "mpnet":
#         mlm_head = AutoModelForMaskedLM.from_pretrained("microsoft/mpnet-base").cuda().eval().lm_head
#     else:
#         mlm_head = AutoModelForMaskedLM.from_pretrained("bert-base-uncased").cuda().eval().cls
#     return mlm_head