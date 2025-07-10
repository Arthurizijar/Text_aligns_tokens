import os
import argparse
import openpyxl

from data import get_text_similarity_data
from embedding import get_embeddings, get_logits
from model import load_model, load_decoder_layer
from metric import get_decode_content, calculate_similarity
from utils import check_args

def evaluate_align_quality(args):
    
    data_name, model_name = args.dataset, args.model
    print(data_name, model_name)
    texts_a, texts_b = get_text_similarity_data(data_name=data_name)
    model, tokenizer, doc_model, doc_tokenizer = load_model(args)
    model_list = [model, tokenizer, doc_model, doc_tokenizer]

    embs_a = get_embeddings(model_name=model_name, model_list=model_list, data_name=data_name, texts=texts_a).cuda()
    if data_name == "msmarco":
        embs_b = get_embeddings(model_name=model_name, model_list=model_list, data_name=data_name,
                                texts=texts_b, isdoc=True).cuda()
    else:
        embs_b = get_embeddings(model_name=model_name, model_list=model_list, data_name=data_name,
                                texts=texts_b).cuda()

    # embedding_layers = get_encode_embedding_weights(model_name, model_list)
    # embedding_layers = get_embedding_weights(model_name=model_name, model=model)
    decoder_layer = load_decoder_layer(args, model=model)
    
    logits_a = get_logits(args, decoder_layer, embs_a)
    logits_b = get_logits(args, decoder_layer, embs_b)
    
    # cos_sim_a, dot_product_a, norm_a, norm_l = calculate_similarity(embeddings=embs_a,
    #                                                                 embeddings_layers=embedding_layers,
    #                                                                 layer2norm=True)
    # cos_sim_b, dot_product_b, norm_b = calculate_similarity(embeddings=embs_b,
    #                                                         embeddings_layers=embedding_layers)

    decoded_tokens_a = get_decode_content(tokenizer=tokenizer, logits=logits_a, k=args.k)
    decoded_tokens_b = get_decode_content(tokenizer=tokenizer, logits=logits_b, k=args.k)
    # info_list = [texts_a, texts_b, cos_tokens_a, cos_tokens_b, dot_tokens_a, dot_tokens_b, norm_a, norm_b, norm_l]

    # mlm_head = load_mlm_head(model_name)
    # dot_product_a = mlm_head(embs_a)
    # dot_product_b = mlm_head(embs_b)
    # cos_tokens_a, dot_tokens_a = get_decode_content(tokenizer=tokenizer, k=k,
    #                                                 dot_products=dot_product_a)

    # cos_tokens_b, dot_tokens_b = get_decode_content(tokenizer=tokenizer, k=k,
    #                                                 dot_products=dot_product_b)

    info_list = [texts_a, texts_b, decoded_tokens_a, decoded_tokens_b]

    return info_list


def save_xlsx(save_path, info_list):
    texts_a, texts_b, decoded_tokens_a, decoded_tokens_b = info_list
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    for index, tuple in enumerate(zip(texts_a, texts_b, decoded_tokens_a, decoded_tokens_b)):
        for j in range(4):
            sheet.cell(row=index + 1, column=j + 1, value=str(tuple[j]))
    workbook.save(save_path)


if __name__ == "__main__":

    # "bert", "prompt_bert", "simcse", "contriever", "dpr", "sgpt_nli", "sgpt_msmarco",
    # "opt_eol", "gte", "e5", "llama_eol_cse", "opt_eol_cse", "e5_mistral", "syncse",
    # "mistral", "llm2vec_mistral", "GritLM"
    
    parse = argparse.ArgumentParser()
    parse.add_argument('--model', type=str, help='the embedder name')
    parse.add_argument('--dataset', type=str, help='the dataset name', choices=["sts", "msmarco"])
    parse.add_argument('--llama_path', type=str, help='the name of dataset', default="/data/LLM/llama1/llama-7b")
    parse.add_argument('--weight_dir', type=str, help='the dir to save the lm_head weight', default="./save/model_weights/")
    parse.add_argument('--output_dir', type=str, help='the dir to save the xlsx file', default="./output/")
    parse.add_argument('--k', type=int, help='the number of aligned token', default=10)
    
    args = parse.parse_args()
    
    check_args(args)
    
    save_path = f"{args.model}_{args.dataset}_results.xlsx"
    if os.path.exists(save_path):
        print(f"Existing file {save_path} in the directory {args.output_dir}")
        exit(0)
    else:
        args.save_path = save_path
    
    info_list = evaluate_align_quality(args)
    save_xlsx(os.path.join(args.output_dir, args.save_path), info_list)