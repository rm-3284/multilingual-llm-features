import argparse
import os
import sys

sae_directory = "/export/home/rmitsuhashi/sae_lens-6.16.0"
sys.path.insert(0, sae_directory)
from sae_lens import SAE


def load_args():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--model', default='gemma-2-2b', type=str,  help='model')
    # parser.add_argument('--model', default='gemma-2-9b', type=str,  help='model')
    # parser.add_argument('--model', default='Meta-Llama-3.1-8B', type=str,  help='model')
    parser.add_argument('--model_path', type=str,  help='model path')

    parser.add_argument('--modified_layer_num', default=1, type=int, choices=[1, 2, 3], help='model')
    parser.add_argument('--start_idx', default=0, type=int,  help='model')
    parser.add_argument('--topk_feature_num', default=3, type=int,  help='model')
    parser.add_argument('--target_lan', default=1, type=int,  help='model')

    
    
    args = parser.parse_args()
    if args.model == 'gemma-2-9b':
        args.layer_num = 42
    elif args.model == 'gemma-2-2b':
        args.layer_num = 26
    elif args.model == 'Meta-Llama-3.1-8B':
        args.layer_num = 32
    
    if args.model_path == '':
        raise ValueError("No valid args.model_path found. Please manually specify the model_path")

    return args


def load_sae(layer, args):
    if 'gemma-2-2b' in args.model_path:
        release = "gemma-scope-2b-pt-res"
    elif 'gemma-2-9b' in args.model_path:
        release = "gemma-scope-9b-pt-res"
    elif 'Meta-Llama-3.1-8B' in args.model_path:
        release = "llama_scope_lxr_8x"

    if 'Llama' in args.model:
        from lm_saes import SparseAutoEncoder
        sae = SparseAutoEncoder.from_pretrained(f"./Llama3_1-8B-Base-LXR-8x/Llama3_1-8B-Base-L{layer}R-8x")
    elif 'gemma' in args.model:
        root_dir = f'/alt/llms/majd/multilingual-llm-features/SAE/{release}/layer_{layer}/width_16k/'
        file_names = list(os.listdir(root_dir))
        file_names.sort(key=lambda x: int(x.split('_')[-1]))
        file_name = file_names[2]
        sae_id = os.path.join(root_dir, file_name).split(f'{release}/')[1]
        sae, cfg_dict, sparsity = SAE.from_pretrained(
            release=release,  # see other options in sae_lens/pretrained_saes.yaml
            sae_id=sae_id,  # won't always be a hook point
            device='cuda',
            )
    return sae
