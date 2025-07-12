import os
from model import load_decoder_layer

def check_args(args):
    
    check_dir_list = [args.weight_dir, args.output_dir]
    if hasattr(args, "save_dir"):
        check_dir_list.append(args.save_dir)
    
    for dir_path in check_dir_list:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
    
    if args.model in ["llama_eol_cse"]:
        assert os.path.exists(args.args.llama_path), "Please assign the local address of llama1-7b to the 'args.llama_path' variable"
    
    elif args.model in ["sgpt_nli", "sgpt_msmarco", "llm2vec_mistral_unsup", "llm2vec_mistral_sup", "e5_mistral"]:
        load_decoder_layer(args)