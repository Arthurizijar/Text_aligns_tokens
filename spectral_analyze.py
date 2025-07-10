import matplotlib.pyplot as plt
import seaborn as sns
from torch import nn
import torch
import pickle
import os
import pickle
import argparse

from model import load_model, load_decoder_layer
from data import get_text_similarity_data
from embedding import get_embeddings, get_logits
from metric import calculate_similarity, get_decode_content
from utils import check_args

default_alpha = {"gpt_neo": -47, "prompt_opt": -31, "prompt_llama": -105, "mistral": 0.59}


def get_top_eigen_vector(data_name, model_name, data_size):
    data_name, model_name, data_size
    texts_a = get_text_similarity_data(data_name=data_name)
    texts_a = texts_a[:data_size]
    model, tokenizer, doc_model, doc_tokenizer = load_model(args, model_name=model_name)
    model_list = [model, tokenizer, doc_model, doc_tokenizer]

    embs_a = get_embeddings(model_name=model_name, model_list=model_list, data_name=data_name, texts=texts_a)
    embs_a = embs_a.cuda()
    U, S, V = torch.svd(embs_a)
    original_coeff = embs_a @ V

    file_path = os.path.join(args.save_dir, f"{model_name}_{data_name}_top1_eigenvec.pth")
    if not os.path.exists(file_path):
        file = open(file_path, "wb")
        pickle.dump(V[:, :1].T, file)

    return V, original_coeff


def get_coeff_after_training(data_name, model_name, eigen_matrix, data_size):

    texts_a = get_text_similarity_data(data_name=data_name)
    texts_a = texts_a[:data_size]
    model, tokenizer, doc_model, doc_tokenizer = load_model(args, model_name=model_name)
    model_list = [model, tokenizer, doc_model, doc_tokenizer]

    embs_a = get_embeddings(model_name=model_name, model_list=model_list, data_name=data_name, texts=texts_a)
    embs_a = embs_a.cuda()
    trained_coeff = embs_a @ eigen_matrix
    return trained_coeff


def calculate_and_decode(tokenizer, embedding, layer, k):
    cos_sim, dot_product, norm, norm = calculate_similarity(embeddings=embedding,
                                                            embeddings_layers=layer,
                                                            layer2norm=True)
    cos_tokens, dot_tokens = get_decode_content(tokenizer=tokenizer, k=k,
                                                cos_similarities=cos_sim,
                                                dot_products=dot_product)
    return cos_tokens, dot_tokens



def get_prob_dist(args, top1_eigenvec):

    data_name, model_name, alpha, text = args.dataset, args.original_model, args.lambda_value, args.text
    model, tokenizer, doc_model, doc_tokenizer = load_model(args, model_name=model_name)
    model_list = [model, tokenizer, doc_model, doc_tokenizer]
    embs_a = get_embeddings(model_name=model_name, model_list=model_list, data_name=data_name, texts=text).cuda()

    embs_a = embs_a + alpha * top1_eigenvec

    decoder_layer = load_decoder_layer(args, model)
    logits = get_logits(args, decoder_layer, embs_a)
    top_K_logits, top_K_indices = torch.topk(logits, args.k)
    aligned_tokens = [tokenizer.convert_ids_to_tokens(idx.item()) for idx in top_K_indices[0]]
    return aligned_tokens, top_K_logits[0].numpy().tolist()


def draw_multibar_figure(x_list, y_list, label_list, path, share_x=False):
    
    sns.set_theme(style="whitegrid")
    f, axs = plt.subplots(len(x_list), 1, figsize=(9*len(x_list)+1, 8), sharex=False)
    if len(x_list) == 1:
        axs = [axs]
    for i, (x, y, label) in enumerate(zip(x_list, y_list, label_list)):
        if share_x and i != 0:
            sorted_combined = sorted(zip(y, x), key=lambda x: x[0], reverse=True)
            y = [item[0] for item in sorted_combined]
            x = [item[1] for item in sorted_combined]
        sns.barplot(x=x, y=y, hue=x, palette="coolwarm", ax=axs[i])
        axs[i].axhline(0, color="k", clip_on=False)
        axs[i].set_ylabel(label)

    leg = plt.legend()
    axs[-1].get_legend().remove()
    plt.savefig(path, bbox_inches='tight', dpi=300)


def draw_figure(data, path):
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(6, 6))
    sns.lineplot(x=range(len(data)), y=data)
    plt.xlabel('$i$ (Dismention)')
    plt.ylabel('$v_i$ (Variation)')
    plt.title('Contribution to the aligned tokens')
    plt.savefig(path, bbox_inches='tight', dpi=300)
    
    
    
def subvec_decode_contribution(args):
    
    data_name, model_name, k, text = args.dataset, args.original_model, args.k, args.text
    
    file_path = os.path.join(args.save_dir, f"{model_name}_{data_name}_top1_eigenvec.pth")
    assert os.path.exists(file_path), "Please run the analysis of `variation` to obtain the first principal component."
    
    model, tokenizer, doc_model, doc_tokenizer = load_model(args, model_name=model_name)
    model_list = [model, tokenizer, doc_model, doc_tokenizer]
    embs_a = get_embeddings(model_name=model_name, model_list=model_list, data_name=data_name, texts=text).cuda()
    
    file = open(file_path, "rb")
    top1_eigenvec = pickle.load(file)
    top_1_subvec = embs_a @ top1_eigenvec.T @ top1_eigenvec
    res_subvec = embs_a - top_1_subvec

    decoder_layer = load_decoder_layer(args, model)
    logits = get_logits(args, decoder_layer=decoder_layer, embs=embs_a)
    logits_1 = get_logits(args, decoder_layer=decoder_layer, embs=top_1_subvec)
    logits_r = get_logits(args, decoder_layer=decoder_layer, embs=res_subvec)
    top_K_similarities, top_K_indices = torch.topk(logits, k)
    each_cos_max_tokens = [tokenizer.convert_ids_to_tokens(idx.item()) for idx in top_K_indices[0]]

    top_K_similarities_1 = logits_1[:, top_K_indices[0]]
    top_K_similarities_r = logits_r[:, top_K_indices[0]]

    figure_path = os.path.join(args.output_dir, f"{model_name}_contribution.png")
    draw_multibar_figure(x_list=[each_cos_max_tokens] * 3,
                         y_list=[top_K_similarities[0].cpu().detach().tolist(),
                                 top_K_similarities_1[0].cpu().detach().tolist(),
                                 top_K_similarities_r[0].cpu().detach().tolist()],
                         label_list=["total", "top1", "non top1"],
                         path=figure_path, share_x=True)
    
    

def before_after_variation(args):
    
    V, original_coeff = get_top_eigen_vector(data_name=args.dataset, model_name=args.original_model, data_size=args.data_size)
    trained_coeff = get_coeff_after_training(data_name=args.dataset, model_name=args.model, eigen_matrix=V, data_size=args.data_size)
    different_value = (trained_coeff - original_coeff).mean(dim=0).cpu().numpy()
    
    file_path = os.path.join(args.save_dir, f"{args.original_model}->{args.model}_{args.dataset}_variation.pth")
    if not os.path.exists(file_path):
        file = open(file_path, "wb")
        pickle.dump(different_value, file)
    
    figure_path = os.path.join(args.output_dir, f"{args.original_model}->{args.model}_{args.dataset}_variation.png")
    draw_figure(different_value, figure_path)
    print(f"The figure has been plotted successfully and saved at {figure_path}")
    

def change_first_component(args):
    if args.lambda_type == "v1":
        file_path = os.path.join(args.save_dir, f"{args.original_model}->{args.model}_{args.dataset}_variation.pth")
        assert os.path.exists(file_path), "Please run the analysis of `variation` to obtain the variation of first principal component."
        file = open(file_path, "rb")
        variation_values = pickle.load(file)
        args.lambda_value = variation_values[0]
    else:
        assert args.lambda_value is not None, "The lambda type has been chosen to be 'custom', --lambda_value is necessary"
    
    file_path = os.path.join(args.save_dir, f"{args.original_model}_{args.dataset}_top1_eigenvec.pth")
    assert os.path.exists(file_path), "Please run the analysis of `variation` to obtain the first principal component."
    file = open(file_path, "rb")
    top1_eigenvec = pickle.load(file)
    aligned_tokens, top_K_logits = get_prob_dist(args, top1_eigenvec=top1_eigenvec)
    
    figure_path = os.path.join(args.output_dir, f"{args.original_model}_change1st_{args.lambda_type}_{args.lambda_value:.2f}.png")
    print([aligned_tokens])
    print([top_K_logits])
    draw_multibar_figure(x_list=[aligned_tokens], y_list=[top_K_logits], label_list=[f"{args.lambda_type} ({args.lambda_value:.2f})"], path=figure_path)
    print(f"The figure has been plotted successfully and saved at {figure_path}")


if __name__ == "__main__":
    
    parse = argparse.ArgumentParser()
    parse.add_argument('--model', type=str, help='the embedder name', default="sgpt_nli")
    parse.add_argument('--original_model', type=str, help='the backbone name', default="gpt_neo")
    parse.add_argument('--dataset', type=str, help='the dataset name', choices=["wiki", "sts", "msmarco"], default="wiki")
    parse.add_argument('--analyze_type', type=str, help="the purpose of spectral analysis", choices=["contribution", "variation", "change"])
    parse.add_argument('--data_size', type=int, help="the data size for SVD", default=100)
    parse.add_argument('--text', type=str, help='the text for analysis', default="YMCA in South Australia")
    parse.add_argument('--llama_path', type=str, help='the name of dataset', default="/data/LLM/llama1/llama-7b")
    parse.add_argument('--weight_dir', type=str, help='the dir to save the lm_head weight', default="./save/model_weights/")
    parse.add_argument('--save_dir', type=str, help='the dir to save the intermediate results', default="./save/spectral_analyze/")
    parse.add_argument('--output_dir', type=str, help='the dir to save the xlsx file', default="./output/spectral_analyze/")
    parse.add_argument('--k', type=int, help='the number of aligned token', default=10)
    parse.add_argument("--lambda_type", type=str, help="the value type of lambda", choices=["v1", "custom"], default="v1")
    parse.add_argument("--lambda_value", type=float, help="the custom value of lambda")
    
    args = parse.parse_args()

    check_args(args)
    
    if args.analyze_type == "variation":
        before_after_variation(args)
    elif args.analyze_type == "contribution":
        subvec_decode_contribution(args)
    elif args.analyze_type == "change":
        change_first_component(args)
        
