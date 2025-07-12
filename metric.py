import os.path
import numpy as np
import torch.nn as nn
import torch


def calculate_similarity(embeddings, embeddings_layers):
    dot_products = embeddings @ embeddings_layers.T
    norm_embeddings = torch.linalg.norm(embeddings, dim=-1)
    norm_layers = torch.linalg.norm(embeddings_layers, dim=-1)
    norm_matrix = norm_embeddings.unsqueeze(1) @ norm_layers.unsqueeze(0)
    cos_similarities = dot_products / norm_matrix

    dot_products = torch.where(torch.isnan(dot_products), torch.full_like(dot_products, -np.inf), dot_products)
    cos_similarities = torch.where(torch.isnan(cos_similarities), torch.full_like(cos_similarities, -1),
                                   cos_similarities)
   
    return cos_similarities, dot_products, norm_embeddings



# def compare_sentence_decode():
#     texts = ["I like apple", "I do not like apple"]
#     for model_name in ["sgpt_nli", "sgpt_msmarco"]:
#         load_model(model_name=model_name)
#         embs_a = get_embeddings(model_name=model_name, data_name='sts', texts=texts)
#         embedding_layers = get_embedding_weights(model_name=model_name)
#         cos_sim_a, dot_product_a, norm_a, norm_l = calculate_similarity(embeddings=embs_a,
#                                                                         embeddings_layers=embedding_layers,
#                                                                         layer2norm=True)
#         cos_tokens_a, dot_tokens_a = get_decode_content(cos_similarities=cos_sim_a, dot_prodoucts=dot_product_a)
#         print(cos_tokens_a)
#         print(dot_tokens_a)



    # except:
    #     print(f"error happens in {data_name} and {model_name}")
