import os
import torch
import argparse
import torch.nn as nn
from tqdm import tqdm
from quantization.quantizer import *
from quantization.bilevel_quantizer import BilevelQuantizer
from quantization.quant_groups import Quantizer
from internvl.model.internvl_chat import InternVLChatConfig, InternVLChatModel


def get_named_linears(module):
    return {name: m for name, m in module.named_modules() if isinstance(m, nn.Linear)}


# core quantization method (simulated quantization)
def pseudo_quantize_tensor(w, n_bit=8,
                           zero_point=True, q_group_size=-1,
                           inplace=False,
                           get_scale_zp=False
                           ):
    org_w_shape = w.shape
    if q_group_size > 0:
        assert org_w_shape[-1] % q_group_size == 0
        w = w.reshape(-1, q_group_size)
    elif q_group_size == -1:
        w = w.reshape(-1, w.shape[-1])
    assert w.dim() == 2
    if zero_point:
        max_val = w.amax(dim=1, keepdim=True)
        min_val = w.amin(dim=1, keepdim=True)
        max_int = 2 ** n_bit - 1
        min_int = 0
        scales = (max_val - min_val).clamp(min=1e-5) / max_int
        zeros = (-torch.round(min_val / scales)).clamp_(min_int, max_int)
    else:  # we actually never used this
        assert min_val is None
        max_val = w.abs().amax(dim=1, keepdim=True)
        max_val = max_val.clamp(min=1e-5)
        max_int = 2 ** (n_bit - 1) - 1
        min_int = - 2 ** (n_bit - 1)
        scales = max_val / max_int
        zeros = 0

    assert torch.isnan(scales).sum() == 0
    assert torch.isnan(w).sum() == 0

    if inplace:
        ((w.div_(scales).round_().add_(zeros)).clamp_(
            min_int, max_int).sub_(zeros)).mul_(scales)
    else:
        w = (torch.clamp(torch.round(w / scales) +
                         zeros, min_int, max_int) - zeros) * scales
    assert torch.isnan(w).sum() == 0

    w = w.reshape(org_w_shape)

    if get_scale_zp:
        return w, scales.view(w.shape[0], -1), zeros.view(w.shape[0], -1)
    else:
        return w


@torch.no_grad()
def pseudo_quantize_model_weight(
    model, w_bit, q_config, quant_type="int", save_path=None, quant_vit=False, quant_llm=True, qq_groupsize=None
):  
    print('quantizing embedding', '!'*100)
    embedding_groupsize = 32
    embedding_bits = 4
    emb = model.language_model.model.embed_tokens
    quantizer = Quantizer()
    quantizer.configure(embedding_bits, perchannel=True, sym=True)
    shape = emb.weight.data.shape
    W = emb.weight.data.to(torch.float16).reshape((-1, embedding_groupsize))
    quantizer.find_params(W, weight=True)
    emb.weight.data = quantizer.quantize_dequantize(W).reshape(shape).to(torch.float)
    model.language_model.model.embed_tokens = emb

    print("quantizing lm_heads", '!'*100)
    lm_head = model.language_model.lm_head
    lm_quantizer = Quantizer()
    lm_quantizer.configure(embedding_bits, perchannel=True, sym=True)
    lm_shape = lm_head.weight.data.shape
    lm_W = lm_head.weight.data.to(torch.float16).reshape((-1, embedding_groupsize))
    lm_quantizer.find_params(lm_W, weight=True)
    lm_head.weight.data = lm_quantizer.quantize_dequantize(lm_W).reshape(lm_shape).to(torch.float)
    model.language_model.lm_head = lm_head

    if save_path is not None:
        # embedding
        save_quant_dict = {"sublayer_name": "embedding",
                            "weight": quantizer.quantize(W),
                            "scale": quantizer.scale}
        full_path = save_path + "/" + "embed_tokens" + "/"
        os.makedirs(full_path, exist_ok=True)
        torch.save(save_quant_dict, full_path + "embed_tokens")

        # lm head
        lm_save_quant_dict = {"sublayer_name": "lm_head",
                            "weight": lm_quantizer.quantize(lm_W),
                            "scale": lm_quantizer.scale}

        lm_full_path = save_path + "/" + "lm_head" + "/"
        os.makedirs(lm_full_path, exist_ok=True)
        torch.save(lm_save_quant_dict, lm_full_path + "lm_head")

        # others
        modules_quant = ["embed_tokens", "lm_head",
                            "qkv", "proj", "fc1", "fc2",
                            "q_proj", "k_proj", "v_proj", "o_proj", "down_proj", "up_proj", "gate_proj"]
        not_quantized_weights = {}
        for name, param in model.named_parameters():
            if not any(module in name for module in modules_quant):
                not_quantized_weights[name] = param
            else:
                if "bias" in name:
                    not_quantized_weights[name] = param
        print('save not_quantized_weights')
        for k,v in not_quantized_weights.items():
            print(k, v.shape)
        torch.save(not_quantized_weights, save_path + "/not_quantized_weights.pt")

    if quant_type == "int2-bilevel" or quant_type == "int3-bilevel" or quant_type == "int4-bilevel":
        quantizer = BilevelQuantizer()
        if quant_type == "int2-bilevel":
            quantizer.configure(bits=2, perchannel=True, sym=False, round_zero=True, qq_scale_bits=4, qq_scale_sym=False, qq_zero_bits=2, qq_groupsize=qq_groupsize, qq_zero_sym=None)
        elif quant_type == "int3-bilevel":
            quantizer.configure(bits=3, perchannel=True, sym=False, round_zero=True, qq_scale_bits=4, qq_scale_sym=False, qq_zero_bits=3, qq_groupsize=qq_groupsize, qq_zero_sym=None)
        else:
            quantizer.configure(bits=4, perchannel=True, sym=False, round_zero=True, qq_scale_bits=4, qq_scale_sym=False, qq_zero_bits=4, qq_groupsize=qq_groupsize, qq_zero_sym=None)
        
        layers = []
        if quant_llm:
            layers.extend(model.language_model.model.layers)
        if quant_vit:
            layers.extend(model.vision_model.encoder.layers)
        
        print(quant_vit, quant_llm)
        print(f"Quantizing {len(layers)} layers...")
        print(layers)
        print("*" * 100)
        
        for i in tqdm(range(len(layers)), desc=f"pseudo {quant_type} weight quantization..."):
            print(f"Quantizing layer {i} / of {len(layers)}...")
            if save_path is not None:
                layer_path = f"{save_path}/{i}"
                os.makedirs(layer_path, exist_ok=True)
            
            named_linears = get_named_linears(layers[i])
            for n, m in named_linears.items():
                original_shape = m.weight.data.shape
                reshaped_weight = m.weight.reshape(original_shape[0],-1, q_config["q_group_size"]).permute(1,0,2).reshape(-1,q_config["q_group_size"])
                quantizer.find_params(reshaped_weight, weight=True)
                m.weight.data = quantizer.quantize_dequantize(reshaped_weight).reshape(-1, original_shape[0], q_config["q_group_size"]).permute(1,0,2).reshape(original_shape)
                quantized_weight = quantizer.quantize(reshaped_weight)
                quantized_weight = quantized_weight.reshape(-1, original_shape[0], q_config["q_group_size"]).permute(1,0,2).reshape(original_shape)
                quantized_weight = quantized_weight.reshape(-1, q_config["q_group_size"])

                # 保存量化信息
                if save_path is not None:
                    quant_dict = {
                        "quant_weights": quantized_weight.to(torch.int8),
                        "quant_layer_scale": quantizer.quant_scale.to(torch.int8) if hasattr(quantizer, "quant_scale") else None,
                        "quant_layer_zeros": quantizer.zero.to(torch.int8),
                        "quant_layer_scale_qq_scale": quantizer.qq_scale.scale if hasattr(quantizer, "qq_scale") else None,
                        "quant_layer_scale_qq_zero": quantizer.qq_scale.zero if hasattr(quantizer, "qq_scale") else None,
                    }
                    torch.save(quant_dict, f"{layer_path}/{n}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument(
        "model_path",
        type=str,
        help="Path to vlm model to load",
    )
    parser.add_argument(
        "--wbits",
        type=int,
        default=16,
        help="#bits to use for quantization; use 16 for evaluating base model.",
    )
    parser.add_argument(
        "--groupsize",
        type=int,
        default=None,
        help="How many weight columns (input features) are quantized with the same statistics, default = all of them",
    )
    parser.add_argument(
        "--qq_groupsize",
        type=int,
        default=None,
        help="qq_groupsize",
    )
    parser.add_argument("--save_merged", type=str, default=False, help="Path to save the merged model")
    parser.add_argument("--save_quant_info", type=str, default=False, help="Path to save quantized statistics.")

    parser.add_argument("--quant_vit", type=str, default="false", help="Whether to quant_vit")
    parser.add_argument("--quant_mlp", type=str, default="false", help="Whether to quant_mlp")
    parser.add_argument("--quant_llm", type=str, default="false", help="Whether to quant_llm")
    
    args = parser.parse_args()

    args.quant_vit = args.quant_vit.lower() == 'true'
    args.quant_mlp = args.quant_mlp.lower() == 'true'
    args.quant_llm = args.quant_llm.lower() == 'true'

    # Load teacher model
    print("\n============ Loading model... ============")
    config = InternVLChatConfig.from_pretrained(args.model_path)
    model = InternVLChatModel.from_pretrained(
          args.model_path, torch_dtype=torch.bfloat16, config=config)

    print("\n============ Quantizing model... ============")
    q_config = {
        "zero_point": True,  # by default True
        "q_group_size": args.groupsize,  # whether to use group quantization
    }

    model = model.cuda()
    print(args.quant_vit, args.quant_llm)
    pseudo_quantize_model_weight(
        model, w_bit=args.wbits, q_config=q_config, quant_type="int{}-bilevel".format(args.wbits), save_path=args.save_quant_info, quant_vit=args.quant_vit, quant_llm=args.quant_llm, qq_groupsize=args.qq_groupsize
    )

    print("\n============ Saving Model Weights... ============")
    if args.save_merged:
        model.save_pretrained(args.save_merged, safe_serialization=False)
