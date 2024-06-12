from model import GPT
from dataclasses import dataclass
import os
import math
import time
import inspect
import torch
import tiktoken
import torch.nn as nn
from torch.nn import functional as F
from hellaswag import render_example, iterate_examples
from topoformer import AttentionHead, LocallyConnected
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

device = "cuda"

ATTENTION_DICT = {}
ATTENTION_DICT["attention"] = []
def hook_fn(module, input, output):
    # Take mean of tokens, should make this a parameter to pass!
    print(f'shape of ourput: {output.shape}')
    ATTENTION_DICT["attention"].append(output[:, -1, :].detach().cpu())
    y = ATTENTION_DICT["attention"]



def _register_hook(model, layer_name):
    for name, layer in model.named_modules():
        if name == layer_name:
            print(f'registering hoook for layer name {layer_name}, name {name}')
            hook = layer.register_forward_hook(hook_fn)
    return hook


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 16
    n_embd: int = 784


ckpt_path = "log/model_19072.pt"
checkpoint = torch.load(ckpt_path, map_location=device)
gptconf = GPTConfig()
model = GPT(gptconf)
state_dict = checkpoint['model']
unwanted_prefix = '_orig_mod.'
for k,v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
model.load_state_dict(state_dict)
LAYER = "transformer.h.4.attn.queries"
hook = _register_hook(model, LAYER)

model.eval()
model.to(device)

enc = tiktoken.get_encoding("gpt2")
_ = model.generate(enc, "hi how are", 200)



attention_data = torch.stack(list(ATTENTION_DICT.values())[0]).squeeze(1)
print(attention_data.shape)



def apply_pca(attention_data, n_components=10, explained_variance_cutoff=5):
        pca = PCA(n_components=n_components)
        transformed_data = pca.fit_transform(attention_data)
        print(f"transformed data shape inside apply pca: {transformed_data.shape}")
        components = pca.components_

        pca_variance = PCA(n_components=explained_variance_cutoff)
        pca_variance.fit(attention_data)
        # transformed data: PCA dimensionality reduces
        transformed_data = transformed_data
        # Weights vectors(Eigen vectors)
        components = components
        variance_explained = pca_variance.explained_variance_ratio_
        return transformed_data, components, pca_variance.explained_variance_ratio_

def plot(data, components, variance):
    titles = [f"PC{i+1} weights" for i in range(components.shape[0])]
    for idx, component in enumerate(components[:2]):
        plt.figure()
        sns.heatmap(component.reshape(28, 28), center=0)
        plt.savefig(f"pc{idx+1}_{LAYER}.png")
        plt.close()

X = attention_data
pca, components, variance = apply_pca(X)
plot(X, components, variance)

"""
layer name is transformer.h.11.attn.c_proj
layer name is transformer.h.11.attn.keys
layer name is transformer.h.11.attn.queries
layer name is transformer.h.11.attn.values
"""