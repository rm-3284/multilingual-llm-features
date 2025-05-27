import os
from matplotlib.ticker import MultipleLocator
from functools import partial
from tqdm import tqdm
import pandas as pd
import torch
import numpy as np
from transformers import AutoModelForCausalLM,  AutoTokenizer
import matplotlib.pyplot as plt
from utils import load_args, load_sae
import scipy.stats as stats



# Ablate varying numbers of features for target language, compute CE loss on all language corpora (head(500)), plot in one comprehensive figure
def change_activation_print_ce_corpus_different_same_feature_diff_lan_all(args):
    # gen_lid_data(args)
    lan_dict = {'en': 'English', 'es': 'Spanish', 'fr': 'French', 'ja': 'Japanese', 'ko': 'Korean', 'pt': 'Portuguese', 'th': 'Thai', 'vi': 'Vietnamese', 'zh': 'Chinese', 'ar': 'Arabic'}
    # my_model = Chat_Model(args.model_path)
    multilingual_data = pd.read_json('./data/multilingual_data_test.jsonl', lines=True)
    multilingual_data = multilingual_data.groupby('lan').head(500)
    results_dict = {}
    target_lan = ['en', 'es', 'fr', 'ja', 'ko', 'pt', 'th', 'vi', 'zh', 'ar'][args.target_lan]
    plt.rcParams.update({
        'font.size': 20,               # Global font size
        'font.weight': 'bold',         # Global font weight (bold)
        'axes.labelweight': 'bold',    # Axis labels
        'axes.titleweight': 'bold',    # Title
    })
    fig, axs = plt.subplots(3, 3, figsize=(18, 20), sharey=True)
    lan_list = ['en', 'es', 'fr', 'ja', 'ko', 'pt', 'th', 'vi', 'zh', 'ar']
    pos_dict = {'fr': 0, 'es': 1, 'ja': 2}
    print(f'original')
    for idx, lan in enumerate(lan_list):
        i = (idx-1)//3
        j = (idx-1) % 3
        if lan not in ['es', 'fr', 'ja', 'ko', 'pt', 'th', 'vi', 'zh', 'ar']:
            continue
        # lan=lan_list[args.target_lan]
        # load_dir = f'./plot/line_chart_ce_loss/{args.model}/{lan}'
        save_dir = f'./plot/line_chart_ce_loss/{args.model}/{target_lan}'
        os.makedirs(save_dir, exist_ok=True)
        ori_ce_loss = np.load(os.path.join(save_dir, 'ori_ce_loss.npy'))
        sae_ce_loss_all_layer_0_2 = np.load(os.path.join(save_dir, f'sae_ce_loss_all_layer_0_2.npy'))-ori_ce_loss
        sae_ce_loss_all_layer_0_1 = np.load(os.path.join(save_dir, f'sae_ce_loss_all_layer_0_1.npy'))-ori_ce_loss
        sae_ce_loss_all_layer_1_1 = np.load(os.path.join(save_dir, f'sae_ce_loss_all_layer_1_1.npy'))-ori_ce_loss
        # sae_ce_loss_all_layer_0_3 = np.load(os.path.join(save_dir, f'sae_ce_loss_all_layer_0_3.npy'))-ori_ce_loss
        # 创建x轴的索引，假设它们是相同的
        x = list(range(len(sae_ce_loss_all_layer_0_2)))

        # plt.figure(figsize=(10, 4))

        marker_styles = ['o', 's', '^', 'D', 'x', '+', '*', 'v', '>', '<']

        ce_loss = sae_ce_loss_all_layer_0_1[:, 500*idx:500*(1+idx)]
        means = np.mean(ce_loss, axis=1)
        line1, = axs[i, j].plot(x, means, label=f'Rank #1 Feature', linestyle='-', linewidth=2, marker='o', color='red')

        ce_loss = sae_ce_loss_all_layer_1_1[:, 500*idx:500*(1+idx)]
        means = np.mean(ce_loss, axis=1)
        line2, = axs[i, j].plot(x, means, label=f'Rank #2 Feature', linestyle='-', linewidth=2, marker='s', color='green')

        ce_loss = sae_ce_loss_all_layer_0_2[:, 500*idx:500*(1+idx)]
        means = np.mean(ce_loss, axis=1)
        line3, = axs[i, j].plot(x, means, label=f'Rank #1 and #2 Features', linestyle='-', linewidth=2, marker='^', color='orange')

        # ce_loss = sae_ce_loss_all_layer_0_3[:, 500*idx:500*(1+idx)]
        # means = np.mean(ce_loss, axis=1)
        # plt.plot(x, means, label=f'Top 1-3 Features', linestyle='-',linewidth=2)

        axs[i, j].set_title(f'CE Loss for {lan_dict[lan]}')
        axs[i, j].set_xlabel('Layer')
        axs[i, j].set_ylim(-0.5, 13)  # Set the y-axis limits
        # axs[i,j].ylabel('Changes in CE Loss')
        axs[i, j].grid(True)

    axs[0, 0].set_ylabel('Changes in CE Loss')
    axs[1, 0].set_ylabel('Changes in CE Loss')
    axs[2, 0].set_ylabel('Changes in CE Loss')

    # plt.legend()
    # plt.legend(ncol=2, loc='upper center', bbox_to_anchor=(0.5, -0.2),frameon=False, handlelength=1.5, handletextpad=0.5)
    fig.legend([line1, line2, line3], [f'Rank #1 {lan_dict[target_lan]} Feature', f'Rank #2  {lan_dict[target_lan]} Feature', f'Rank #1 and #2  {lan_dict[target_lan]} Features'],
               loc='upper center', ncol=3, bbox_to_anchor=(0.5, 0.10), frameon=False,
               handlelength=1.5, handletextpad=0.5)


    # plt.show()
    plt.tight_layout(rect=[0, 0.1, 1, 0.95])
    plt.savefig(os.path.join(save_dir, f'line_chart_ce_loss_{args.model}_all.pdf'), format='pdf', bbox_inches='tight')
    plt.savefig(os.path.join(save_dir, f'line_chart_ce_loss_{args.model}_all.png'), format='png', bbox_inches='tight')
    plt.close()
    # with open(f'./change_results/{args.model}-{int(time.time())}.json', 'w') as json_file:


# Ablate specific number of features for target language, compute CE loss on both target and other languages (first 500 samples), plot two separate lines
def change_activation_print_ce_corpus(args):
    # gen_lid_data(args)
    lan_dict = {'en': 'English', 'es': 'Spanish', 'fr': 'French', 'ja': 'Japanese', 'ko': 'Korean', 'pt': 'Portuguese', 'th': 'Thai', 'vi': 'Vietnamese', 'zh': 'Chinese', 'ar': 'Arabic'}
    # my_model = Chat_Model(args.model_path)
    multilingual_data = pd.read_json('./data/multilingual_data_test.jsonl', lines=True)
    multilingual_data = multilingual_data.groupby('lan').head(500)
    results_dict = {}
    lan_list = ['en', 'es', 'fr', 'ja', 'ko', 'pt', 'th', 'vi', 'zh', 'ar']
    print(f'original')
    for idx, lan in enumerate(lan_list):
        # lan=lan_list[args.target_lan]
        save_dir = f'./plot/line_chart_ce_loss/{args.model}/{lan}'
        os.makedirs(save_dir, exist_ok=True)
        ori_ce_loss = np.load(os.path.join(save_dir, 'ori_ce_loss.npy'))
        sae_ce_loss_all_layer = np.load(os.path.join(save_dir, f'sae_ce_loss_all_layer_{args.start_idx}_{args.topk_feature_num}.npy'))
        ce_difference_all_layer = sae_ce_loss_all_layer-ori_ce_loss

        x = list(range(len(ce_difference_all_layer)))
        plt.rcParams.update({
            'font.size': 20,               # Global font size
            'font.weight': 'bold',         # Global font weight (bold)
            'axes.labelweight': 'bold',    # Axis labels
            'axes.titleweight': 'bold',    # Title
        })
        plt.figure(figsize=(10, 4))

        ce_loss = ce_difference_all_layer[:, 500*idx:500*(1+idx)]
        means = np.mean(ce_loss, axis=1)
        confidence_interval = 0.99
        n = ce_difference_all_layer.shape[1]
        sem = stats.sem(ce_difference_all_layer, axis=1)  
        h = sem * stats.t.ppf((1 + confidence_interval) / 2, n-1)  

        plt.plot(x, means, label=f'{lan_dict[lan]}', linestyle='-', linewidth=2)
        # plt.errorbar(x, means, fmt='none', yerr=h, ecolor='black', elinewidth=1, capsize=3)

        ce_loss = np.concatenate((ce_difference_all_layer[:, :500*idx], ce_difference_all_layer[:, 500*(1+idx):]), axis=1)
        means = np.mean(ce_loss, axis=1)
        confidence_interval = 0.99
        n = ce_difference_all_layer.shape[1]
        sem = stats.sem(ce_difference_all_layer, axis=1)  
        h = sem * stats.t.ppf((1 + confidence_interval) / 2, n-1)  

        plt.plot(x, means, label='Other Languages', linestyle='-', linewidth=2)
        # plt.errorbar(x, means, fmt='none', yerr=h, ecolor='black', elinewidth=1, capsize=3)

        plt.title(f'Ablating {lan_dict[lan]} Features')
        plt.xlabel('Layer')
        plt.ylabel('Changes in CE Loss')
        # plt.yticks(np.linspace(10, 35, 6))
        y_major_locator = MultipleLocator(3)
        ax = plt.gca()

        ax.yaxis.set_major_locator(y_major_locator)

        plt.grid(True)

        # plt.legend()
        plt.legend(ncol=2, loc='upper center', bbox_to_anchor=(0.5, -0.2), frameon=False, handlelength=1.5, handletextpad=0.5)


        # plt.show()
        plt.savefig(os.path.join(save_dir, f'line_chart_ce_loss_{args.model}_{lan}_{args.start_idx}_{args.topk_feature_num}.pdf'), format='pdf', bbox_inches='tight')
        plt.savefig(os.path.join(save_dir, f'line_chart_ce_loss_{args.model}_{lan}_{args.start_idx}_{args.topk_feature_num}.png'), format='png', bbox_inches='tight')
        plt.close()
        # with open(f'./change_results/{args.model}-{int(time.time())}.json', 'w') as json_file:
