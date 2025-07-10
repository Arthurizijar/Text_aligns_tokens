import os
from model import load_decoder_layer

def check_args(args):
    for dir_path in [args.weight_dir, args.save_dir, args.output_dir]:
        if dir_path is None:
            continue
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
    
    if args.model in ["llama_eol_cse"]:
        assert os.path.exists(args.args.llama_path), "Please assign the local address of llama1-7b to the 'args.llama_path' variable"
    
    elif args.model in ["sgpt_nli", "sgpt_msmarco", "llm2vec_mistral_unsup", "llm2vec_mistral_sup", "llm2vec_llama2", "llm2vec_llama2_sup", "e5_mistral"]:
        load_decoder_layer(args)