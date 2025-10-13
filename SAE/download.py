# 下载数据集
import os
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from huggingface_hub import snapshot_download
from huggingface_hub import login
# access_token_read = 'YOUR_OWN_ACESS_TOKEN'
# login(token = access_token_read)
#download_path = "./Llama3_1-8B-Base-LXR-8x"
#snapshot_download(repo_id="fnlp/Llama3_1-8B-Base-LXR-8x", local_dir=download_path)

# only download part of SAE that are actually used
allow_patterns = ["layer_*/width_16k/*", "embedding/*", ".gitattributes", "README.md", "LICENSE"]
download_path = "./gemma-scope-2b-pt-res"
snapshot_download(repo_id="google/gemma-scope-2b-pt-res", local_dir=download_path, allow_patterns=allow_patterns)

#download_path = "./gemma-scope-9b-pt-res"
#snapshot_download(repo_id="google/gemma-scope-9b-pt-res", local_dir=download_path)

