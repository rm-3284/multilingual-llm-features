import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import json
import matplotlib.pyplot as plt
from scipy import stats
from tqdm import tqdm
import pandas as pd
import argparse
from transformers import AutoModelForCausalLM,  AutoTokenizer
import numpy as np
import torch
from utils import load_args, load_sae



torch.set_grad_enabled(False)  # avoid blowing up mem


# Sorted indices based on descending values of the monolingual metric (absolute activation differences)
def generate_top_index_magnitude(args):
    for layer in tqdm(range(args.layer_num)):
        # layer=0
        file_dir = f'./sae_acts/{args.model}/layer_{layer}/'
        all_sae_acts = torch.load(os.path.join(file_dir, 'sae_acts.pth'))
    
        num_lan = len(all_sae_acts)//100
        all_sae_acts_per_token = []
        for acts in all_sae_acts:
            all_sae_acts_per_token.append(acts[0, 1:, :])
        avg_act_per_lan = []
        for i in range(num_lan):
            all_sae_acts_per_token_lan = torch.concat(all_sae_acts_per_token[100*i:100*(i+1)])
            avg_act = all_sae_acts_per_token_lan.mean(dim=-2)
            avg_act_per_lan.append(avg_act)
        avg_act_per_lan = torch.stack(avg_act_per_lan)
        top_index_per_lan = []
        top_ratio_per_lan = []
        for i in range(num_lan):
            avg_act_difference_per_lan=avg_act_per_lan[i]-torch.cat([avg_act_per_lan[:i], avg_act_per_lan[i+1:]], dim=0).mean(dim=0)
            sorted_values, sorted_indices=torch.sort(avg_act_difference_per_lan, descending=True)
            top_ratio_per_lan.append(sorted_values.unsqueeze(0))
            top_index_per_lan.append(sorted_indices.unsqueeze(0))
        top_index_per_lan = torch.concat(top_index_per_lan)
        top_ratio_per_lan = torch.concat(top_ratio_per_lan)
        torch.save(top_index_per_lan, os.path.join(file_dir, 'top_index_per_lan_magnitude.pth'))


if __name__ == "__main__":
    args = load_args()
    generate_top_index_magnitude(args)
