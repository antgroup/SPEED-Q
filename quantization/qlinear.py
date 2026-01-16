import torch
from torch import Tensor, device, dtype, nn
from quantization.quantizer import *
from quantization.bilevel_quantizer import BilevelQuantizer

# mix_quant_4bit = ["q_proj", "k_proj", "v_proj", "o_proj", "down_proj", "qkv", "proj"]
mix_quant_4bit = ["q_proj", "k_proj", "v_proj", "o_proj", "down_proj"]

def convertModelToQuant(model, 
                        modules_to_not_convert=["embed_tokens", "lm_head", "1", "3"], 
                        current_key_name=None, 
                        has_been_replaced=False,
                        compute_dtype=torch.bfloat16, 
                        quant_type="clsq-n2f3", 
                        q_group_size=128,
                        qq_group_size=None,
                        mix_quant=False):
    
    for name, module in model.named_children():
        if current_key_name is None:
            current_key_name = []
        current_key_name.append(name)
        if (isinstance(module, nn.Linear)) and name not in modules_to_not_convert:
            print("Convert {} ...".format(name))
            in_features = module.in_features
            out_features = module.out_features
            weight = module.weight
            bias = module.bias
            
            if not mix_quant:
                model._modules[name] = QLinear(
                    in_features,
                    out_features,
                    module.bias is not None,
                    compute_dtype=compute_dtype,
                    quant_type=quant_type,
                    q_group_size=q_group_size,
                    qq_group_size=qq_group_size
                )
            else:
                if name in mix_quant_4bit:
                    print("Convert {} to 4bit".format(name))
                    new_quant_type = "int4-bilevel"
                    new_q_group_size = 32
                else:
                    new_quant_type = quant_type
                    new_q_group_size = q_group_size
                model._modules[name] = QLinear(
                    in_features,
                    out_features,
                    module.bias is not None,
                    compute_dtype=compute_dtype,
                    quant_type=new_quant_type,
                    q_group_size=new_q_group_size,
                    qq_group_size=qq_group_size
                )

            model._modules[name].weight = weight
            model._modules[name].bias = bias
            has_been_replaced = True
            # Store the module class in case we need to transpose the weight later
            model._modules[name].source_cls = type(module)
        if len(list(module.children())) > 0:
            _, has_been_replaced = convertModelToQuant(
                module,
                modules_to_not_convert,
                current_key_name,
                has_been_replaced,
                compute_dtype,
                quant_type,
                q_group_size,
                qq_group_size,
                mix_quant
            )
        # Remove the last key for recursion
        current_key_name.pop(-1)
    return model, has_been_replaced


class QLinear(nn.Linear):
    def __init__(self, input_features, output_features, bias=True, compute_dtype=torch.bfloat16, quant_type="ste-n2f3", q_group_size=128, qq_group_size=128, device=None):
        super().__init__(input_features, output_features, bias, device)

        if quant_type == "ste-n2f3":
            self.weight_quantizer = SteN2F3Quantizer(q_group_size=q_group_size)
        elif quant_type == "int2-asym":
            self.weight_quantizer = SteInt2AsymQuantizer(q_group_size=q_group_size)
        elif quant_type == "int3-asym":
            self.weight_quantizer = SteInt3AsymQuantizer(q_group_size=q_group_size)
        elif quant_type == "int3-bilevel":
            self.weight_quantizer = BilevelQuantizer()
            self.weight_quantizer.configure(bits=3, perchannel=True, sym=False, round_zero=True, qq_scale_bits=4, qq_scale_sym=False, qq_zero_bits=3, qq_groupsize=qq_group_size, qq_zero_sym=None)
        elif quant_type == "int2-bilevel":
            self.weight_quantizer = BilevelQuantizer()
            self.weight_quantizer.configure(bits=2, perchannel=True, sym=False, round_zero=True, qq_scale_bits=4, qq_scale_sym=False, qq_zero_bits=2, qq_groupsize=qq_group_size, qq_zero_sym=None)
        elif quant_type == "int4-bilevel":
            self.weight_quantizer = BilevelQuantizer()
            self.weight_quantizer.configure(bits=4, perchannel=True, sym=False, round_zero=True, qq_scale_bits=4, qq_scale_sym=False, qq_zero_bits=4, qq_groupsize=qq_group_size, qq_zero_sym=None)
        else:
            raise ValueError(f"Has no support {quant_type}. Valid quant_type:[ste-n2f3, int2-asym]")
        self.quant_type = quant_type
        self.compute_dtype = compute_dtype
        self.q_group_size = q_group_size
        self.qq_group_size = qq_group_size
        # for resume load checkpoint
        if self.quant_type == "int3-bilevel" or self.quant_type == "int2-bilevel" or self.quant_type == "int4-bilevel":
            # change reshape weights
            weight_shape = self.weight.data.shape
            reshaped_weight = self.weight.reshape(weight_shape[0],-1, self.q_group_size).permute(1,0,2).reshape(-1,self.q_group_size)
            self.weight_quantizer.find_params(reshaped_weight, weight=True)

    def forward(self, x: torch.Tensor):
        if self.bias is not None and self.bias.dtype != x.dtype:
            self.bias.data = self.bias.data.to(x.dtype)

        inp_dtype = x.dtype

        if self.compute_dtype is not None:
            x = x.to(self.compute_dtype)

        bias = None if self.bias is None else self.bias.to(self.compute_dtype)
        out = None

        if self.quant_type == "int3-bilevel" or self.quant_type == "int2-bilevel" or self.quant_type == "int4-bilevel":
            original_shape = self.weight.data.shape
            reshaped_weight = self.weight.reshape(original_shape[0],-1, self.q_group_size).permute(1,0,2).reshape(-1,self.q_group_size)
            self.weight_quantizer.find_params(reshaped_weight, weight=True)
            quantize_weight = self.weight_quantizer.quantize_dequantize(reshaped_weight).reshape(-1, original_shape[0], self.q_group_size).permute(1,0,2).reshape(original_shape)
            quantize_weight = quantize_weight.to(self.compute_dtype)
        else:
            quantize_weight = self.weight_quantizer(self.weight.to(self.compute_dtype))
        
        out = F.linear(x, quantize_weight, bias).to(inp_dtype)

        return out
