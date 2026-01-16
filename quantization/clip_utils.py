import os
import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

import sys
sys.path.append("../")
from internvl.model.internvl_chat import InternVLChatConfig, InternVLChatModel


def get_calib_dataset(datasets="pileval", tokenizer=None, n_samples=128, block_size=1024):
    if datasets == "pile":
        return get_pile_dataset(tokenizer=tokenizer, n_samples=n_samples, block_size=block_size)


def get_pile_dataset(tokenizer=None, n_samples=512, block_size=512):
    dataset = load_dataset("json", data_files="/xnncloud/public/yuanxi/LLM/datasets/mit-han-lab/pile-val-backup/val.jsonl.zst", split="train")
    dataset = dataset.map(lambda x: {
        'text': x['text']
    })
    dataset = dataset.shuffle(seed=42)
    samples = []
    n_run = 0

    for data in dataset:
        line = data["text"]
        line = line.strip()
        line_encoded = tokenizer.encode(line)
        if len(line_encoded) > 512:
            continue
        sample = torch.tensor([line_encoded])
        if sample.numel() == 0:
            continue
        samples.append(sample)
        n_run += 1
        if n_run == n_samples:
            break
    # now concatenate all samples and split according to block size
    cat_samples = torch.cat(samples, dim=1)
    n_split = cat_samples.shape[1] // block_size
    print(f" * Split into {n_split} blocks")
    return [cat_samples[:, i*block_size:(i+1)*block_size] for i in range(n_split)]


def get_blocks(model):
    layers = model.model.layers
    return layers


def append_str_prefix(x, prefix):
    if isinstance(x, str):
        return prefix + x
    elif isinstance(x, tuple):
        return tuple([append_str_prefix(y, prefix) for y in x])
    elif isinstance(x, list):
        return [append_str_prefix(y, prefix) for y in x]
    else:
        return x


def move_embed(model, device):
    model.model.embed_tokens = model.model.embed_tokens.to(device)


def get_named_linears(module):
    return {name: m for name, m in module.named_modules() if isinstance(m, nn.Linear)}


def get_op_by_name(module, op_name):
    # get the op by its name relative to the module
    for name, m in module.named_modules():
        if name == op_name:
            return m
    raise ValueError(f"Cannot find op {op_name} in module {module}")


def get_op_name(module, op):
    # get the name of the op relative to the module
    for name, m in module.named_modules():
        if m is op:
            return name
    raise ValueError(f"Cannot find op {op} in module {module}")


def build_model_and_enc(model_path):
    if not os.path.exists(model_path):  # look into ssd
        raise FileNotFoundError(f"{model_path} not found!")
    print(f"* Building model {model_path}")

    # hf model
    config = InternVLChatConfig.from_pretrained(model_path)
    model = InternVLChatModel.from_pretrained(
          model_path, torch_dtype=torch.bfloat16, config=config)

    enc = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # change
    model = model.language_model.cuda()
    
    model.eval()

    return model, enc


@torch.no_grad()
def apply_clip(module, clip_results):
    if not isinstance(clip_results, list):
        clip_list = clip_results["clip"]
    else:
        clip_list = clip_results
    if len(clip_list) == 0:
        return
    if len(clip_list[0]) == 3:
        for name, max_val, min_val in clip_list:
            layer = get_op_by_name(module, name)
            layer.cuda()
            max_val = max_val.to(layer.weight.device)
            min_val = min_val.to(layer.weight.device)
            org_shape = layer.weight.shape
            layer.weight.data = layer.weight.data.reshape(*max_val.shape[:2], -1)
            layer.weight.data = torch.clamp(layer.weight.data, min_val, max_val)
            layer.weight.data = layer.weight.data.reshape(org_shape)
    else:
        raise 1