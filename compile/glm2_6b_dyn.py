import os
import tvm
import onnx
import torch
import shutil
import argparse
from tvm import relay
from memory_profiler import profile
from tvm.relay.dataflow_pattern import *
from tvm.relay.frontend.common import infer_shape
from tvm.driver.tvmc.model import TVMCModel
from tvm.contrib.hexaflake.aigpu_utils import (
    hexaflake_aigpu_mod_rewrite,
    hexaflake_aigpu_bfloat16_func_register,
)
from tvm.contrib.hexaflake.relay.transform.rewrite import (
    ChangeBatchWithReshape,
    CompressMatmul,
)
from tvm.contrib.hexaflake.relay.transform.incremental import (
    IncrementalAttention,
    LLM_CONFIGS,
)


hexaflake_aigpu_bfloat16_func_register()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, help="the path of source model.")
    parser.add_argument("--target-path", type=str, default=".", help="the path of compiled model.")
    parser.add_argument("--static-batch-size", type=int, default=1, help="the static batch size of model.")
    parser.add_argument("--total-len", type=int, default=2048, help="the total tokens of model infer")
    parser.add_argument("--pre-len", type=int, default=256, help="the max input tokens of base model.")
    parser.add_argument("--is-bf16", type=bool, default=True,  help="the target dtype of compile model.")
    parser.add_argument("--quant", type=bool, default=True, help="int8 quantization of the model.")
    parser.add_argument("--int4",  type=bool, default=False, help="int4 quantization of the model.")
    parser.add_argument("--ngf",  type=bool, default=True, help="compile the model with suffix 'ngf', otherwise, only 'tar'.")
    parser.add_argument("--export-name", type=str, default="glm2_6b2048_48M", help="export ngf|tar model name")

    return parser.parse_args()


def get_cos_sin_table(max_len=32768, rotary_dim=64, base=10000, dtype=torch.float32):
    theta = 1.0 / (base ** (torch.arange(0, rotary_dim, 2, dtype=dtype) / rotary_dim))
    seq_idx = torch.arange(max_len, dtype=dtype)
    idx_theta = torch.outer(seq_idx, theta).float()
    cos_table, sin_table = torch.cos(idx_theta)[:, None, None, :].numpy(), torch.sin(idx_theta)[:, None, None, :].numpy()
    cos_table, sin_table = relay.const(cos_table, dtype="float32"), relay.const(sin_table, dtype="float32")
    return cos_table, sin_table


class BroadcastSimplifier(DFPatternCallback):
    def __init__(self):
        super().__init__()
        self.input = wildcard()
        reshape = is_op("expand_dims")(self.input)
        self.broadcast = is_op("broadcast_to")(reshape)
        self.reshape = is_op("reshape")(self.broadcast)
        self.reshape1 = is_op("reshape")(self.reshape)
        self.transpose = is_op("transpose")(self.reshape1)
        self.transpose1 = is_op("transpose")(self.transpose)
        self.pattern = is_op("nn.batch_matmul")(wildcard(), self.transpose1)

    def callback(self, pre, post, node_map):
        input = node_map[self.input][0]
        reshape1 = node_map[self.reshape1][0]
        transpose = node_map[self.transpose][0]
        transpose1 = node_map[self.transpose1][0]
        data = post.args[0]
        data_shape = infer_shape(pre.args[0])
        data = relay.reshape(
            data, [data_shape[0] // 16, 16 * data_shape[1], data_shape[2]]
        )
        newshape = list(reshape1.attrs.newshape)
        newshape[1] = newshape[1] // 16
        bmm = relay.nn.batch_matmul(
            data,
            relay.transpose(
                relay.transpose(relay.reshape(input, newshape), transpose.attrs.axes),
                transpose1.attrs.axes,
            ),
            transpose_a=pre.attrs.transpose_a,
            transpose_b=pre.attrs.transpose_b,
        )
        return relay.reshape(bmm, infer_shape(pre))


class RotarySimplifier(DFPatternCallback):
    def __init__(self, is_bf16):
        super().__init__()
        self.input = wildcard()
        self.reshape = is_op("reshape")(self.input)
        self.take0 = is_op("take")(self.reshape, is_constant())
        self.take1 = is_op("take")(self.reshape, is_constant())
        self.pos_tab0 = is_constant()
        self.pos_tab1 = is_constant()
        self.index = wildcard()
        self.emb0 = is_op("annotate.view")(
            is_op("reshape")(is_op("take")(self.pos_tab0, self.index))
        )
        emb1 = is_op("annotate.view")(
            is_op("reshape")(is_op("take")(self.pos_tab1, wildcard()))
        )
        mul0 = is_op("multiply")(self.take0, self.emb0) - is_op("multiply")(
            self.take1, emb1
        )
        emb0 = is_op("annotate.view")(
            is_op("reshape")(is_op("take")(self.pos_tab0, wildcard()))
        )
        emb1 = is_op("annotate.view")(
            is_op("reshape")(is_op("take")(self.pos_tab1, wildcard()))
        )
        take1 = is_op("take")(self.reshape, is_constant())
        take0 = is_op("take")(self.reshape, is_constant())
        mul1 = is_op("multiply")(take1, emb0) + is_op("multiply")(take0, emb1)
        concat = is_op("concatenate")(
            is_tuple([is_op("expand_dims")(mul0), is_op("expand_dims")(mul1)])
        )
        self.pattern = concat
        self.pos_emb = None
        self.is_bf16 = is_bf16

    def callback(self, pre, post, node_map):
        src_shape = list(pre.checked_type.shape)
        input = node_map[self.input][0]
        pos_tab0 = node_map[self.pos_tab0][0]
        pos_tab1 = node_map[self.pos_tab1][0]
        index = node_map[self.index][0]
        emb0 = node_map[self.emb0][0]
        emb_shape = list(emb0.attrs.real_shape)
        emb_shape[-1] = emb_shape[-1] * 4
        if self.pos_emb is None:
            pos_tab = relay.reshape(
                relay.concatenate(
                    (
                        relay.expand_dims(pos_tab0, -1),
                        relay.expand_dims(-pos_tab1, -1),
                        relay.expand_dims(pos_tab1, -1),
                        relay.expand_dims(pos_tab0, -1),
                    ),
                    axis=-1,
                ),
                (0, -1, emb_shape[-1]),
            ) * relay.const(2.0)
            self.pos_emb = relay.annotation.view(
                relay.reshape(
                    relay.take(pos_tab, index, axis=0),
                    (emb_shape[0] * emb_shape[1], 1, 1, emb_shape[-1]),
                ),
                (emb_shape[0] * emb_shape[1], 1, 1, emb_shape[-1]),
                (),
            )
        if self.is_bf16:
            input = relay.annotation.view(input, infer_shape(input), ())
        branch = relay.reshape(input, (-3, -1, emb_shape[-1] // 4, 2))
        if self.is_bf16:
            branch = relay.annotation.view(branch, infer_shape(branch), (1, 1, 32, 2))
        branch = relay.concatenate([branch, branch], -1)
        if self.is_bf16:
            branch = relay.annotation.view(branch, infer_shape(branch), (1, 1, 32, 2))
        branch = relay.reshape(
            branch, (src_shape[0] * src_shape[1], -1, 1, emb_shape[-1])
        )
        branch = branch * self.pos_emb
        return relay.nn.avg_pool2d(branch, (1, 2,), (1, 2,), count_include_pad=True)


class SoftmaxRewrite(DFPatternCallback):
    def __init__(self):
        super().__init__()
        self.input_x = wildcard()
        self.pattern = is_op("nn.softmax")(self.input_x)

    def callback(self, pre, post, node_map):
        input = node_map[self.input_x][0]
        data_shape = infer_shape(pre.args[0])
        softmax_dim = pre.attrs.axis
        if data_shape.__len__() >= 4 and data_shape[softmax_dim] > 4096:
            reshaped_input = relay.reshape(input, newshape=([-1, data_shape[softmax_dim]]))
            softmax_out = relay.nn.softmax(reshaped_input, axis=-1)
            reshaped_out = relay.reshape(softmax_out, data_shape)
            return reshaped_out
        return post


def simplify_5d_ops(mod, is_bf16):
    mod["main"] = rewrite([BroadcastSimplifier(), RotarySimplifier(is_bf16), SoftmaxRewrite()], mod["main"])
    return mod


@profile
def compile_glm():
    args = get_args()

    batch_size = args.static_batch_size
    is_bf16 = args.is_bf16
    quant = args.quant
    total_len = args.total_len
    pre_len = args.pre_len
    int4 = args.int4
    ngf = args.ngf
    export_name = args.export_name
    max_len = total_len
    if total_len > 8192:
        total_len = 2048

    model_path = args.model_path
    onnx_model = onnx.load(model_path)
    mod_o, _ = relay.frontend.from_onnx(
        onnx_model,
        {
            "input_ids": ([1, total_len], "int32"),
            "position_ids": relay.const([list(range(total_len))], "int64"),
            "attention_mask": ([1, total_len], "int64"),
        },
        dtype="int32",
    )

    del onnx_model

    if quant:
        last_dense = mod_o["main"].body[0].args[0].args[0]
        mod_o = CompressMatmul("uint4" if int4 else "int8", excluded=[last_dense])(
            mod_o
        )
    mod = tvm.IRModule.from_expr(mod_o["main"])
    del mod_o
    if batch_size != 1:
        mod = ChangeBatchWithReshape(
            {mod["main"].params[0]: 0, mod["main"].params[1]: 0}, batch_size
        )(mod)

    path = os.path.join(args.target_path, export_name)
    if batch_size != 1:
        path += "_b" + str(batch_size)
    if is_bf16:
        path += "_bf16"
    else:
        path += "_fp32"
    if quant:
        path += "_q"
        if int4:
            path += "4"
    if pre_len != 256:
       path += f"_pre{pre_len}"

    if os.path.isdir(path):
        shutil.rmtree(path)
    os.makedirs(path)

    path = os.path.join(path, path)
    if max_len > 8192:
        ext_emb0, ext_emb1 = get_cos_sin_table(max_len=max_len)
    else:
        ext_emb0, ext_emb1 = None, None

    def glm2_ext_pos_embed(
        expr: relay.Expr, orig: relay.Expr, pos_var: relay.Var
    ) -> relay.Expr:
        def mutate_embed(
            expr: relay.Expr, orig: relay.Expr, pos_var: relay.Var, pos_emb: relay.Expr
        ):
            if isinstance(orig.args[1], relay.Constant):
                org_shape = list(infer_shape(orig.args[0]))
                arg0_shape = list(infer_shape(expr.args[0]))
                arg1_shape = list(infer_shape(expr.args[1]))

                if len(arg0_shape) == len(arg1_shape):
                    for i, val in enumerate(arg1_shape):
                        s = arg0_shape[i]
                        if val == total_len and s == pre_len and org_shape[i] == val:
                            pos = pos_var
                            embedding = pos_emb
                            if i < 1:
                                pos = relay.reshape(pos_var, [-1])
                                pos = relay.annotation.view(
                                    pos, (pre_len * batch_size,), ()
                                )
                                embedding = relay.take(embedding, pos, axis=i)
                                arg1_shape[i] = pre_len
                                arg1_shape[1] = batch_size
                                embedding = relay.reshape(embedding, arg1_shape)
                                embedding = relay.annotation.view(
                                    embedding, arg1_shape, ()
                                )
                                return relay.Call(
                                    orig.op, [expr.args[0], embedding], expr.attrs,
                                )
            return None

        if (
            isinstance(expr.args[0], relay.Call)
            and expr.args[0].op.name == "multiply"
            and isinstance(expr.args[1], relay.Call)
            and expr.args[1].op.name == "multiply"
        ):
            if orig.op.name == "add":
                lhs = mutate_embed(expr.args[0], orig.args[0], pos_var, ext_emb1)
                rhs = mutate_embed(expr.args[0], orig.args[0], pos_var, ext_emb0)
                if lhs and rhs:
                    return lhs + rhs
                return None
            else:
                assert orig.op.name == "subtract"
                lhs = mutate_embed(expr.args[0], orig.args[0], pos_var, ext_emb0)
                rhs = mutate_embed(expr.args[0], orig.args[0], pos_var, ext_emb1)
                if lhs and rhs:
                    return lhs - rhs
                return None
        return None

    inc_cfg = LLM_CONFIGS["ChatGLM2-6B"]
    if max_len > 8192:
        inc_cfg["position_embedding_op"] = ["add", "subtract"]
        inc_cfg["pos_embed_func"] = glm2_ext_pos_embed
    inc = IncrementalAttention(
        inc_cfg,
        "bfloat16" if is_bf16 else "float32",
        length=pre_len,
        batch_size=batch_size,
        max_length=max_len,
    )

    fallback_device = tvm.device("aigpu")
    configs = {
        "relay.fallback_device_type": fallback_device.device_type,
        "relay.SimplifyInference.ignored_ops": "nn.layer_norm;nn.gelu",
        "aigpu.backend.config": {
            "l2_reserved": 12,
            "compressed_pqdense_weight_dtype": "uint4" if int4 else "int8",
            "num_of_devs": 1,
        },
    }
    if is_bf16:
        configs.update({"relay.ToMixedPrecision.mixed_precision_type": "bfloat16"})
        configs.update({"relay.ToMixedPrecision.keep_orig_output_dtype": False})
    target = {"aigpu": "aigpu", "cpu": "llvm"}

    mod_t = inc(mod)
    mod_t = simplify_5d_ops(mod_t, is_bf16)
    print(mod_t)
    tvmc_mod = TVMCModel(mod_t, {})
    mod_t = hexaflake_aigpu_mod_rewrite(mod_t, is_bf16, True, True)
    with tvm.transform.PassContext(opt_level=3, config=configs):
        graph_mod = relay.build(mod_t, target=target, dev_id=-1)

    tvmc_mod.export_package(graph_mod, path + ".tar", "", "", "tar")
    if ngf:
        tvmc_mod.export_package(
            graph_mod, path + ".ngf", "", inc.cross_options, "ngf",
        )
    else:
        with open("base.json", "w") as fo:
            fo.write(inc.cross_options)

    del graph_mod
    del mod_t

    # ------------ compile base incremental ------------
    pre_len = 1
    inc = IncrementalAttention(
        inc_cfg, "bfloat16" if is_bf16 else "float32", length=1, batch_size=batch_size, max_length=max_len
    )
    mod = inc(mod)
    mod = simplify_5d_ops(mod, is_bf16)
    mod = hexaflake_aigpu_mod_rewrite(mod, is_bf16, True, True)

    with tvm.transform.PassContext(opt_level=3, config=configs):
        graph_mod = relay.build(mod, target=target, dev_id=-1)
    path += "_inc"
    tvmc_mod.export_package(graph_mod, path + ".tar", "", "", "tar")
    if ngf:
        tvmc_mod.export_package(
            graph_mod, path + ".ngf", "", inc.cross_options, "ngf",
        )
    else:
        with open("inc.json", "w") as fo:
            fo.write(inc.cross_options)


if __name__ == "__main__":
    compile_glm()
