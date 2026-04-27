from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from freekv import adapter
import torch
from tqdm import tqdm
import json, os, argparse, random
from datasets import load_dataset
import numpy as np
import time

c = torch.cuda.get_device_capability()
os.environ["TORCH_CUDA_ARCH_LIST"] = f"{c[0]}.{c[1]}"

BOLD   = "\033[1m"
RED    = "\033[91m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
RESET  = "\033[0m"
SEP    = "─" * 100

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


def build_chat(tokenizer, prompt, model_name):
    model_name_lower = model_name.lower()
    if "ds-r1" in model_name_lower:
        return prompt + "<think>\n"
    if "llama" in model_name_lower:
        messages = [
            {"role": "user", "content": f"{prompt}"}
        ]
        return tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to("cuda")
    if "qwen" in model_name_lower:
        messages = [
            {"role": "system", "content": "You are a helpful and harmless assistant. You are Qwen developed by Alibaba. You should think step-by-step."},
            {"role": "user", "content": prompt}
        ]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    if "qwq" in model_name_lower:
        messages = [
            {"role": "user", "content": prompt}
        ]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    return prompt


def generate_once(model, input_ids, max_gen, temperature, eos_token_ids, pad_token_id):
    if temperature > 0:
        return model.generate(
            input_ids,
            max_new_tokens=max_gen,
            num_beams=1,
            do_sample=True,
            temperature=temperature,
            eos_token_id=eos_token_ids,
            pad_token_id=pad_token_id,
            past_key_values=None,
        )
    return model.generate(
        input_ids,
        max_new_tokens=max_gen,
        do_sample=False,
        eos_token_id=eos_token_ids,
        pad_token_id=pad_token_id,
        past_key_values=None,
    )


def simplify_text_preview(text, max_tokens=100):
    tokens = text.split()
    return " ".join(tokens[:max_tokens])


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--model", type=str, default=None, help="Model name (key in config/model2path.json)")
    parser.add_argument("--dataset", type=str, required=True, choices=["AIME24", "gov_report", "lgbench"], help="Evaluation dataset to use")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (0 for greedy)")
    parser.add_argument("--max_length", type=int, default=32000, help="Max input token length (longer inputs are truncated from the middle)")
    parser.add_argument("--max_gen", type=int, default=8192, help="Max number of new tokens to generate")
    parser.add_argument("--data_idx", type=int, default=None, help="Single data index to evaluate")
    parser.add_argument("--data_idx_to", type=int, default=None, help="Evaluate data from index 0 to this index (exclusive)")

    parser.add_argument("--sink", type=int, default=512, help="Number of sink tokens to keep")
    parser.add_argument("--recent", type=int, default=512, help="Number of recent tokens to keep")
    parser.add_argument("--budget", type=int, default=2048, help="Total token budget including sink and recent")
    parser.add_argument("--page_size", type=int, default=32, help="KV cache retrieval page size")
    parser.add_argument("--cpu_layout", type=str, default="HND", choices=["NHD", "HND"], help="CPU KV cache memory layout")
    parser.add_argument("--spec_ret", action="store_true", help="Enable speculative retrieval")
    parser.add_argument("--repeat_bsz", type=int, default=1, help="Repeat input to simulate batch size")
    parser.add_argument("--thread_pool", type=int, default=2, help="Number of threads in the recall thread pool")
    parser.add_argument("--n_recall_stream", type=int, default=2, help="Number of CUDA streams for recall")
    parser.add_argument("--recall_impl", type=str, default="cuda_cpy",
                        choices=["arkvale", "torch_cpy", "cuda_cpy"], help="Recall implementation")
    parser.add_argument("--corr", type=float, default=None, help="Correction threshold (cosine similarity); None to disable")
    parser.add_argument("--warmup", type=int, default=2, help="Number of warmup generation rounds before timing")
    parser.add_argument("--log_dir", type=str, default=None, help="If set, write decode/recall CSV logs + meta/gen JSON here")
    parser.add_argument("--run_tag", type=str, default="", help="Tag identifying the run config (e.g. full_kv | freekv | thoughtfetch)")
    parser.add_argument("--ema_alpha", type=float, default=0.1, help="EMA smoothing factor for cos-sim trajectory classifier")
    parser.add_argument("--r_threshold", type=float, default=0.92, help="EMA cos-sim above this is classified as R (steady)")
    parser.add_argument("--t_threshold", type=float, default=0.75, help="EMA cos-sim below this is classified as T (transition)")
    parser.add_argument("--segment_window", type=int, default=16, help="Re-evaluate segment label every N decode steps")
    parser.add_argument("--classifier_layer", type=int, default=None, help="Which transformer layer's cos-sim drives the classifier (default: middle layer)")

    return parser.parse_args(args)


def load_model_and_tokenizer(path):
    dev = torch.device("cuda:0")
    dtype = torch.bfloat16
    tokenizer = AutoTokenizer.from_pretrained(
        path, trust_remote_code=True, use_fast=False
    )
    generation_config = GenerationConfig.from_pretrained(path)
    eos_token_ids = generation_config.eos_token_id
    if not isinstance(eos_token_ids, list):
        eos_token_ids = [eos_token_ids]
    model = (
        AutoModelForCausalLM
        .from_pretrained(path, torch_dtype=dtype, device_map=dev)
        .eval()
    )
    page_size = args.page_size
    token_budgets = args.budget
    page_budgets = token_budgets // page_size
    n_sink_pages = args.sink // page_size
    n_win_pages = args.recent // page_size
    print(f"\n{CYAN}{SEP}")
    print(f"  KV Cache Config: token_budget={token_budgets}, page_budget={page_budgets}, "
          f"page_size={page_size}, sink={args.sink}, recent={args.recent}")
    print(f"{SEP}{RESET}\n")
    if token_budgets > 0:
        infer_state = adapter.enable_offload(
            model,
            dtype=dtype,
            device=dev,
            page_size=page_size,
            page_budgets=page_budgets,
            page_topks=page_budgets-1,
            n_sink_pages=n_sink_pages,
            n_win_pages=n_win_pages,
            n_max_bytes=6 * (1 << 30),
            n_max_cpu_bytes=20 * (1 << 30),
            group_size=1,
            cpu_layout=args.cpu_layout,
            spec_ret=args.spec_ret,
            thread_pool_size=args.thread_pool,
            n_recall_stream=args.n_recall_stream,
            recall_impl=args.recall_impl,
            corr=args.corr,
            log_dir=args.log_dir,
            run_tag=args.run_tag,
            ema_alpha=args.ema_alpha,
            r_threshold=args.r_threshold,
            t_threshold=args.t_threshold,
            segment_window=args.segment_window,
            classifier_layer=args.classifier_layer,
        )
    else:
        assert not args.spec_ret
        infer_state = adapter.enable_offload(
            model,
            dtype=dtype,
            device=dev,
            page_size=page_size,
            page_budgets=None, # page_budgets=None means "full" (no eviction & recall)
            n_max_bytes=6 * (1 << 30),
            n_max_cpu_bytes=1 * (1 << 30),  # CPU pool unused when page_budgets=None
            group_size=1,
            recall_impl=args.recall_impl,
            log_dir=args.log_dir,
            run_tag=args.run_tag,
            ema_alpha=args.ema_alpha,
            r_threshold=args.r_threshold,
            t_threshold=args.t_threshold,
            segment_window=args.segment_window,
            classifier_layer=args.classifier_layer,
        )

    return model, tokenizer, eos_token_ids, infer_state


def get_pred(
    model,
    tokenizer,
    eos_token_ids,
    data,
    answer_field_id,
    max_length,
    max_gen,
    prompt_format,
    model_name,
    temperature,
    warmup,
    infer_state=None,
):
    preds = []
    for prompt_idx, json_obj in enumerate(tqdm(data)):
        if prompt_format is not None:
            prompt = prompt_format.format(**json_obj)
        else:
            prompt = json_obj["prompt"]
        tokenized_prompt = tokenizer(
            prompt, truncation=False, return_tensors="pt"
        ).input_ids[0]
        if len(tokenized_prompt) > 8192 and len(tokenized_prompt) < max_length:
            # only for long inputs
            tokenized_prompt = tokenized_prompt.repeat(max_length // len(tokenized_prompt) + 1)
            assert len(tokenized_prompt) > max_length
        if len(tokenized_prompt) > max_length:
            half = int(max_length / 2)
            prompt = tokenizer.decode(
                tokenized_prompt[:half], skip_special_tokens=True
            ) + tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
        
        chat_prompt = build_chat(tokenizer, prompt, model_name)
        if isinstance(chat_prompt, str):
            input = tokenizer(chat_prompt, truncation=False, return_tensors="pt").to("cuda").input_ids
        else:
            input = chat_prompt
        prompt_length = input.shape[-1]
        print(f"{CYAN}[Input] prompt_length={prompt_length}, dtype={input.dtype}, device={input.device}{RESET}")
        input = input.repeat(args.repeat_bsz, 1)
        with torch.no_grad():
            if warmup > 0:
                print(f"{YELLOW}[Warmup] Running {warmup} warmup round(s)...{RESET}")
                if infer_state is not None:
                    infer_state.is_warmup = True
                for _ in range(warmup):
                    _ = generate_once(
                        model, input, max_gen, temperature, eos_token_ids, tokenizer.eos_token_id
                    )
                if infer_state is not None:
                    infer_state.is_warmup = False
                print(f"{YELLOW}[Warmup] Done.{RESET}")
                model.tbt_stat_ms.clear()

            if infer_state is not None:
                infer_state.prompt_id = prompt_idx
            st = time.perf_counter()
            output = generate_once(
                model, input, max_gen, temperature, eos_token_ids, tokenizer.eos_token_id
            )
            ed = time.perf_counter()
        
        gen_token = [len(o) - len(i) for o, i in zip(output, input)]
        decode_ms = model.tbt_stat_ms[1:]
        avg_tbt = sum(decode_ms) / len(decode_ms) if decode_ms else 0.0

        print(f"\n{GREEN}{SEP}")
        print(f"  [{model_name}] Generation Summary")
        print(f"{SEP}{RESET}")
        print(f"  Tokens generated : {gen_token}")
        print(f"  Total time       : {sum(model.tbt_stat_ms)/1000:.2f}s")
        print(f"  Decode time      : {sum(decode_ms)/1000:.2f}s")
        print(f"  Avg TBT (decode) : {avg_tbt:.2f} ms")
        print(f"{GREEN}{SEP}{RESET}\n")
        for b in range(len(output)):
            pred = tokenizer.decode(output[b], skip_special_tokens=True)
            pred_only_output = tokenizer.decode(output[b][len(input[b]):], skip_special_tokens=True)
            print(f"{BOLD}[Batch {b}] Output preview:{RESET} {simplify_text_preview(pred_only_output, max_tokens=100)}")
            entry = {
                "prompt_id": prompt_idx,
                "batch_idx": b,
                "input": prompt,
                "pred": pred,
                "pred_only_output": pred_only_output,
                "answer": json_obj[answer_field_id] if answer_field_id is not None else "",
                "input_len": len(input[b]),
                "output_len": len(output[b]),
                "gen_tokens": gen_token[b],
                "wall_time_s": ed - st,
                "avg_tbt_ms": avg_tbt,
                "decode_time_ms": sum(decode_ms),
            }
            preds.append({"input:": prompt, "pred": pred,
                           "answer": entry["answer"],
                           "input_len": entry["input_len"],
                           "output_len": entry["output_len"]})
            if infer_state is not None and infer_state.logger.enabled:
                infer_state.logger.append_gen(entry)
    return preds


if __name__ == "__main__":
    args = parse_args()
    seed_everything(args.seed)
    model2path = json.load(open("config/model2path.json", "r"))

    model_name = args.model
    max_length = args.max_length
    max_gen = args.max_gen
    model, tokenizer, eos_token_ids, infer_state = load_model_and_tokenizer(model2path[model_name])

    dataset = args.dataset
    ds_dir = "eval/datasets"
    answer_field_id = None
    prompt_format = None
    if dataset == "gov_report":
        # data_idx: 59, length:32726
        ds_path = f"{ds_dir}/gov_report.jsonl"
        prompt_format = "You are given a report by a government agency. Write a one-page summary of the report.\n\nReport:\n{context}\n\nNow, write a one-page summary of the report.\n\nSummary:"
    elif dataset == "lgbench":
        ds_path = f"{ds_dir}/longgenbench.json"
    elif dataset == "AIME24":
        answer_field_id = "answer"
        ds_path = f"{ds_dir}/aime_2024.jsonl"
        dataset2prompt = json.load(open("eval/o1/config/dataset2prompt.json", "r"))
        prompt_format = dataset2prompt[dataset]

    with open(ds_path, "r") as f:
        if ds_path.endswith(".jsonl"):
            records = [json.loads(line) for line in f if line.strip()]
        else:
            records = json.load(f)
            if not isinstance(records, list):
                records = [records]
    from datasets import Dataset
    data = Dataset.from_list(records)

    if args.data_idx is not None:
        data = data.select(range(args.data_idx, args.data_idx+1))
    elif args.data_idx_to is not None:
        data = data.select(range(0, args.data_idx_to))

    if infer_state is not None and infer_state.logger.enabled:
        try:
            import subprocess
            git_commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=os.path.dirname(__file__) or "."
            ).decode().strip()
        except Exception:
            git_commit = None
        infer_state.logger.write_meta({
            "run_tag": args.run_tag,
            "args": vars(args),
            "model_name": model_name,
            "model_path": model2path[model_name],
            "dataset": dataset,
            "n_prompts": len(data),
            "git_commit": git_commit,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "gpu": torch.cuda.get_device_name(0),
            "torch_version": torch.__version__,
        })

    preds = get_pred(
        model,
        tokenizer,
        eos_token_ids,
        data,
        answer_field_id,
        max_length,
        max_gen,
        prompt_format,
        model_name,
        args.temperature,
        args.warmup,
        infer_state=infer_state,
    )

    if infer_state is not None:
        infer_state.logger.close()

    out_dir = f"tmp_res/{model_name}"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    out_path = f"{out_dir}/{dataset}.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for pred in preds:
            json.dump(pred, f, ensure_ascii=False)
            f.write("\n")


# Original ArkVale
# python eval/o1/pred.py --model ds-r1-qwen-7b --dataset gov_report --temperature 0.0 --max_gen 1024 --data_idx 0 --warmup 0 --recall_impl arkvale

# Without correction
# python eval/o1/pred.py --model ds-r1-qwen-7b --dataset gov_report --temperature 0.0 --max_gen 1024 --data_idx 0 --warmup 0 --recall_impl cuda_cpy --cpu_layout HND --spec_ret

# With correction
# python eval/o1/pred.py --model ds-r1-qwen-7b --dataset gov_report --temperature 0.0 --max_gen 1024 --data_idx 0 --warmup 0 --recall_impl cuda_cpy --cpu_layout HND --spec_ret --corr 0.9
