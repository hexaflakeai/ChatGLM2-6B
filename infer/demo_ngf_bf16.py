from transformers import AutoTokenizer

import numpy as np
import shutil
import argparse
import os
import json
import hxrt as rt
import pickle as pk

from src.hx_infexec import GLMInfer

cur_path = os.path.dirname(os.path.realpath(__file__))

inc = 1
total = 2048
pre = 256
align = 256
output_shape_1 = (1, pre, 130528)
output_shape_2 = (1, inc, 130528)

dev_ids = []
dev_dram_limit = []
dump_golden = 0
split_stragety = 0
engine_version = 0


def check_ret(ret, msg):
    if ret != 0:
        print(msg)
        exit(1)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--engine",
        type=str,
        default="./glm2_6b2048_bf16_multi.ngf",
        help="the compiled engine file to run.",
    )
    parser.add_argument(
        "--base-length",
        default=256,
        type=int,
        help="the base length infer in base model.",
    )
    parser.add_argument(
        "--static-batch-size", default=1, type=int, help="static batch size"
    )
    parser.add_argument(
        "--dynamic-batch-size", default=1, type=int, help="dynamic batch size"
    )
    parser.add_argument(
        "--question-from",
        default=1,
        type=int,
        help="select the source of you questions, 1: user-generated, 2: self-generated.",
    )
    parser.add_argument(
        "--dump-result",
        default="",
        type=str,
        help="serialization result and save it to the path of dump-result.",
    )
    parser.add_argument(
        "--question-index",
        default=0,
        type=int,
        help="choose a question from default inputs by quesiton index",
    )
    parser.add_argument(
        "--do-sample", action="store_true", default=False, help="do sample"
    )
    parser.add_argument("--temperature", default=0.85, type=float, help="temperature")
    parser.add_argument("--top-p", default=1, type=float, help="top p")
    parser.add_argument(
        "--device", action="store", type=str, default="", help="set the device id"
    )
    parser.add_argument(
        "--multi-dev-dram-limit",
        action="store",
        type=str,
        default="",
        help="set dram limit for multi card",
    )
    parser.add_argument(
        "--dump-golden",
        action="store_true",
        default=False,
        help="dump golden to current directory",
    )
    parser.add_argument(
        "--split-stragety",
        action="store",
        type=int,
        default=0,
        help="set model split strategy at multi-card, 0: GREEDY 1: EQUAL 2: LOOSE",
    )
    parser.add_argument("--max-batch-size", default=1, type=int, help="max batch size")
    parser.add_argument(
        "--engine-version",
        default=0,
        type=int,
        help="choose engine version, default is version 0",
    )
    parser.add_argument("--total-len", default=2048, type=int, help="total len")
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=2048,
        help="incremental model max new tokens",
    )
    parser.add_argument(
        "--constant-output",
        type=bool,
        default=False,
        help="keep outputing until the max_new_tokens",
    )

    return parser.parse_args()


class Graph(object):
    def __init__(
        self,
        engine,
        tokenizer_path,
        static_batch_size,
        batch_size,
        input_index,
        max_batch_size,
        base_length,
        constant_output=False,
    ):
        global engine_version, dev_dram_limit, dump_golden
        self.static_batch_size = static_batch_size
        self.dynamic_batch_size = batch_size
        self.base_length = base_length
        self.engine = engine

        self.model = GLMInfer(
            engine,
            batch_count=1,
            static_batch_size=static_batch_size,
            batch_size=batch_size,
            in_out_nparray=True,
            use_cache=True,
            max_batch_size=max_batch_size,
            dev_ids=dev_ids,
            dev_dram_limit=dev_dram_limit,
            dump_golden=dump_golden,
            split_stragety=split_stragety,
            config_file=os.path.join(tokenizer_path, "config.json"),
            base_length=base_length,
            engine_version=engine_version,
            constant_output=constant_output,
            total=total,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path, trust_remote_code=True
        )
        self.input_index = input_index

    def graph_run(
        self, input_data, max_new_tokens=2048, temperature=0.85, top_p=1, do_sample=True, dump_result="",
    ):
        input_data = [f"[Round {1}]\n\n问：{i}\n\n答：" for i in input_data]
        input_data_ = []
        max_len = 0
        input_idx = self.input_index
        for job in range(self.dynamic_batch_size):
            for i in range(self.static_batch_size):
                input_data_.append(input_data[input_idx])
                input_ids = self.tokenizer(input_data[input_idx])["input_ids"]
                max_len = max(max_len, len(input_ids))

                input_idx = input_idx + 1
                if input_idx >= len(input_data):
                    input_idx = 0

        base_round = max_len // self.base_length
        m = max_len % self.base_length
        if m != 0:
            base_round += 1
        print(
            "base_length {}, max_length {}, base_round {}".format(
                self.base_length, max_len, base_round
            )
        )

        total_pre = base_round * self.base_length
        inputs = []
        input_idx = self.input_index
        base_dynamic_batch = 0
        for job in range(self.dynamic_batch_size):
            batch_input = []
            tokens = self.tokenizer(
                input_data_[
                    base_dynamic_batch : base_dynamic_batch + self.static_batch_size
                ],
                padding="max_length",
                max_length=total_pre,
            )
            input_ids = np.array(tokens["input_ids"]).astype("int32")
            batch_size, input_ids_seq_length = input_ids.shape[0], input_ids.shape[-1]
            batch_input.append(input_ids)
            pos = np.array(tokens["position_ids"]).astype("int32")
            batch_input.append(pos)
            attention_mask = np.ones((batch_size, total, total))
            attention_mask = np.tril(attention_mask)
            zero_mask = np.ones((batch_size, total))
            pad_mask = np.array(tokens["attention_mask"])
            zero_mask[:batch_size, : pad_mask.shape[1]] = pad_mask[:batch_size, :]
            attention_mask = attention_mask * np.expand_dims(zero_mask, axis=1)
            attention_mask -= np.expand_dims(zero_mask, axis=-1) - 1
            attention_mask = (attention_mask < 0.5).astype("bool")
            attention_mask = np.expand_dims(attention_mask, axis=1)
            batch_input.append(attention_mask)
            inputs.append(batch_input)
            base_dynamic_batch += self.static_batch_size

        outputs = self.model.inference_use_cache(
            inputs,
            max_new_tokens,
            base_round=base_round,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
        )

        res = []
        index = 0
        batch_count = 0
        for job in range(self.dynamic_batch_size):
            for s in range(self.static_batch_size):
                out_ids = outputs[batch_count][job][s]
                res.append(out_ids)
                answer = self.tokenizer.decode(out_ids)
                print(f"Question {index + 1}:")
                print(f"{input_data_[index]}\n")
                print(f"Response:")
                print(f"{answer}\n")
                index = index + 1
                if index >= len(input_data_):
                    break

        if dump_result != "":
            with open(dump_result, "wb") as f:
                pk.dump(res, f)
        self.model.show_throughput()


def main():
    args = get_args()

    static_batch_size = args.static_batch_size
    dynamic_batch_size = args.dynamic_batch_size
    engine = args.engine
    max_batch_size = args.max_batch_size
    input_index = args.question_index
    base_length = args.base_length
    max_new_tokens = args.max_new_tokens
    temperature = args.temperature
    top_p = args.top_p
    do_sample = args.do_sample
    global engine_version
    engine_version = args.engine_version
    global total
    total = args.total_len
    constant_output = args.constant_output

    ret, count = rt.device_get_count()
    check_ret(ret, "Failed to get device count")

    dev_ids_str = args.device.split(",")
    if dev_ids_str[0] != "":
        for tmp_str in dev_ids_str:
            dev_ids.append(int(tmp_str))
            valid = 0 if (dev_ids[-1] < count and dev_ids[-1] >= 0) else 1
            check_ret(valid, f"Device Id:{dev_ids[-1]} is Invalid")

    dev_dram_limit_str = args.multi_dev_dram_limit.split(",")
    if dev_dram_limit_str[0] != "":
        for tmp_str in dev_dram_limit_str:
            dev_dram_limit.append(int(tmp_str) * 1024 * 1024 * 1024)
    if len(dev_dram_limit) > 0 and len(dev_dram_limit) != len(dev_ids):
        print(f"dev_dram_limit size is not equal size of dev_ids ")
        exit(1)

    global dump_golden
    dump_golden = 1 if args.dump_golden == True else 0
    global split_stragety
    split_stragety = args.split_stragety

    if len(dev_ids) == 1 and dev_ids[0] >= 0:
        rt.set_device(dev_ids[0])

    print("cur_path", cur_path)
    question_file = os.path.join(cur_path, "src", "question.txt")
    default_questions = []
    with open(question_file, "r", encoding="utf-8") as f:
        default_questions = [line.strip() for line in f.readlines() if len(line) > 1]

    c_data = []
    question_file = os.path.join(cur_path, "src", "c.txt")
    with open(question_file, "r", encoding="utf-8") as f:
        c = json.load(f)
        for i in c["question"]:
            c_data.append(i)

    if args.question_from == 1:  # user-generated
        input_data = c_data
    elif args.question_from == 2:  # self-generated
        input_data = default_questions
    tokenizer_path = os.path.join(cur_path, "src", "chatglm2-6b")
    g = Graph(
        engine,
        tokenizer_path,
        static_batch_size,
        dynamic_batch_size,
        input_index,
        max_batch_size,
        base_length,
        constant_output=constant_output,
    )
    g.graph_run(
        input_data,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=do_sample,
        dump_result=args.dump_result,
    )


if __name__ == "__main__":
    question_1 = "你好"
    main()
