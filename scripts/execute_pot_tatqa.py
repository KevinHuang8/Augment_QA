"""
POT (Program of Thought) baseline for TatQA dataset.
Adapted from execute_pot_finqa.py.
"""

import time
import json
import argparse
import copy
import os
import random
import pyrootutils
pyrootutils.setup_root(search_from=__file__, indicator=".project-root", pythonpath=True)

import func_timeout
from typing import List
import platform
import multiprocessing
from tqdm import tqdm
from transformers import AutoTokenizer
from tenacity import retry, stop_after_attempt, wait_exponential
from openai import OpenAI

from utils.tatqa_metric import TaTQAEmAndF1
from utils.utils import load_data_split

ROOT_DIR = os.path.join(os.path.dirname(__file__), "../")

VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")


def safe_execute(code_string: str, keys=None):
    def execute(x):
        try:
            exec(x)
            locals_ = locals()
            if keys is None:
                return locals_.get('ans', 'Error: No variable named "ans"')
            else:
                return locals_.get(keys, None)
        except Exception as e:
            return "Error: " + str(e)
    try:
        ans = func_timeout.func_timeout(5, execute, args=(code_string,))
    except func_timeout.FunctionTimedOut:
        ans = "Error: Timeout"
    return ans


def parse_api_result(result):
    to_return = []
    for idx, g in enumerate(result.choices):
        text = g.message.content
        text = text.replace('```python\n', '').replace('```Python\n', '').replace('```PYTHON\n', '').replace('```', '').strip()
        to_return.append(text)
    return to_return


@retry(stop=stop_after_attempt(6), wait=wait_exponential(multiplier=1, min=4, max=30))
def call_llm_api(engine, messages, max_tokens, temperature, top_p, n, stop, key):
    client = OpenAI(
        base_url=VLLM_BASE_URL,
        api_key="dummy"
    )
    result = client.chat.completions.create(
        model=engine,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        n=n,
        stop=stop,
    )
    return result


def linearize_tatqa_table(data_item, n_rows=1000, max_paragraphs=None):
    """Linearize a TatQA table with paragraph context."""
    lines = []

    if 'document_input' in data_item and data_item['document_input']:
        paragraphs = data_item['document_input']
        if max_paragraphs is not None:
            paragraphs = paragraphs[:max_paragraphs]
        lines.append("Report:")
        lines.append(' '.join(paragraphs))
        lines.append("")

    lines.append("Table:")
    table = data_item['table']
    lines.append(' | '.join(table['header']).strip())
    for row in table['rows'][:n_rows]:
        lines.append(' | '.join(row).strip())

    lines.append(f"\nQ: {data_item['question']}\n")
    return '\n'.join(lines)


TATQA_POT_EXAMPLES = """Read the following report and table, then write a Python program to answer the question.

Report:
Sales by Contract Type: Substantially all of our contracts are fixed-price type contracts. Sales included in Other contract types represent cost plus and time and material type contracts. Total sales for 2019 were $1,496.5 million.

Table:
 | 2019 | 2018 | 2017
Fixed Price | $1,452.4 | $1,146.2 | $1,036.9
Other | 44.1 | 56.7 | 70.8
Total | $1,496.5 | $1,202.9 | $1,107.7

Q: What is the change in total sales from 2018 to 2019?
total_2019 = 1496.5
total_2018 = 1202.9
ans = total_2019 - total_2018


Read the following report and table, then write a Python program to answer the question.

Report:
The company reported revenue growth across all segments. Operating expenses increased primarily due to higher personnel costs and marketing spend.

Table:
 | 2019 | 2018
Revenue | 500.0 | 420.0
Operating expenses | 350.0 | 290.0
Net income | 150.0 | 130.0

Q: What is the percentage change in revenue from 2018 to 2019?
revenue_2019 = 500.0
revenue_2018 = 420.0
ans = (revenue_2019 - revenue_2018) / revenue_2018 * 100


Read the following report and table, then write a Python program to answer the question.

Report:
The following table shows the breakdown of employees by region as of December 31, 2019 and 2018.

Table:
Region | 2019 | 2018
North America | 12000 | 11500
Europe | 8500 | 8200
Asia Pacific | 6300 | 5900
Total | 26800 | 25600

Q: How many regions had more than 8000 employees in 2019?
north_america = 12000
europe = 8500
asia_pacific = 6300
count = 0
if north_america > 8000: count += 1
if europe > 8000: count += 1
if asia_pacific > 8000: count += 1
ans = count


Read the following report and table, then write a Python program to answer the question.

Report:
Lease costs are reported in cost of revenues and operating expenses. The weighted average remaining lease term is 5.2 years.

Table:
 | 2019 | 2018
Operating lease cost | 25.3 | 22.1
Finance lease cost | 3.8 | 4.2
Short-term lease cost | 1.5 | 1.0
Total lease cost | 30.6 | 27.3

Q: What is the ratio of operating lease cost to total lease cost in 2019?
operating_2019 = 25.3
total_2019 = 30.6
ans = operating_2019 / total_2019"""


def worker_annotate(
        pid: int,
        args,
        g_eids: List,
        dataset,
        tokenizer
):
    """A worker process for annotating."""
    with open(args.api_config_file, 'r') as f:
        keys = [line.strip() for line in f.readlines()]

    em_and_f1 = TaTQAEmAndF1()
    g_dict = dict()
    total_num, correct_num = 0, 0

    pbar = tqdm(g_eids, desc=f"POT-TatQA Worker {pid}", position=pid, leave=True)
    for idx, g_eid in enumerate(pbar):
        g_data_item = dataset[g_eid]
        g_dict[g_eid] = {
            'generations': [],
            'ori_data_item': copy.deepcopy(g_data_item)
        }

        few_shot_prompt = TATQA_POT_EXAMPLES

        generate_prompt = '\nRead the following report and table, then write a Python program to answer the question:\n'
        max_row = 100
        n_paragraphs = len(g_data_item.get('document_input', []))
        max_paras = n_paragraphs
        query_table = linearize_tatqa_table(g_data_item, n_rows=max_row, max_paragraphs=max_paras)
        max_prompt_tokens = args.max_api_total_tokens - args.max_generation_tokens
        while len(tokenizer.tokenize(query_table)) >= max_prompt_tokens - 2000:
            if max_row > 10:
                max_row -= 10
            elif max_paras > 1:
                max_paras -= 1
            else:
                break
            query_table = linearize_tatqa_table(g_data_item, n_rows=max_row, max_paragraphs=max_paras)
        generate_prompt += query_table
        prompt = generate_prompt

        prompt_text = prompt
        while len(tokenizer.tokenize(prompt_text)) >= max_prompt_tokens:
            parts = few_shot_prompt.split("\n\n\nRead the following")
            if len(parts) <= 1:
                break
            few_shot_prompt = "Read the following" + "\n\n\nRead the following".join(parts[1:])
            prompt = generate_prompt
            prompt_text = prompt

        prompt += '\nWrite a python program to answer the question. Save the answer in a variable named "ans". Do not output anything else other than the python code.'

        messages = [
            {"role": "user", "content": prompt}
        ]
        result = call_llm_api(
            args.engine,
            messages,
            max_tokens=args.max_generation_tokens,
            temperature=0.0,
            top_p=1,
            n=1,
            stop=['\n\n'],
            key=keys[pid % len(keys)]
        )
        codes = parse_api_result(result)
        error_msg = ''
        r = codes[0]
        if 'ans =' in r or 'ans=' in r:
            ans_key = 'ans'
        else:
            ans_key = r.split('\n')[-1].split('=')[0].strip()
        ans = safe_execute(r, keys=ans_key)
        if isinstance(ans, str) and ans.startswith('Error'):
            error_msg = ans
        g_dict[g_eid]['generations'].append(r)
        if isinstance(ans, set):
            ans = list(ans)
        if not isinstance(ans, list):
            ans = [ans]

        g_dict[g_eid]['pred_answer'] = ans

        gold_answer = g_data_item['answer_text']
        if len(gold_answer) == 2 and gold_answer[-1] == '###':
            gold_answer = gold_answer[0]

        ground_truth = {
            'answer': gold_answer,
            'answer_type': g_data_item['answer_type'],
            'scale': g_data_item['scale'],
            'answer_from': 'table-text',
        }
        score = em_and_f1(ground_truth=ground_truth, prediction=ans, pred_scale='')
        g_dict[g_eid]['score'] = score
        g_dict[g_eid]['error_msg'] = error_msg
        g_dict[g_eid]['answer_type'] = g_data_item['answer_type']
        g_dict[g_eid]['scale'] = g_data_item['scale']
        if score == 1:
            correct_num += 1
        total_num += 1
        pbar.set_postfix(acc=f"{correct_num}/{total_num} ({correct_num / total_num:.2%})")

    overall_em, overall_f1, scale_score, _ = em_and_f1.get_overall_metric()
    print(f"\nWorker {pid} TatQA metrics: EM={overall_em:.4f}, F1={overall_f1:.4f}, Scale={scale_score:.4f}")
    return g_dict


def main():
    args.api_config_file = os.path.join(ROOT_DIR, args.api_config_file)
    args.save_dir = os.path.join(ROOT_DIR, args.save_dir)
    os.makedirs(args.save_dir, exist_ok=True)

    start_time = time.time()
    dataset = load_data_split(args.dataset, args.dataset_split)
    dataset = dataset.select(range(min(args.max_sample_num, len(dataset))))

    generate_eids = list(range(len(dataset)))
    generate_eids_group = [[] for _ in range(args.n_processes)]
    for g_eid in generate_eids:
        generate_eids_group[(int(g_eid) + random.randrange(args.n_processes)) % args.n_processes].append(g_eid)

    print(f'\n******* Running POT on TatQA ({len(dataset)} examples) *******')
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=args.engine)

    g_dict = dict()
    if args.n_processes == 1:
        res = worker_annotate(0, args, generate_eids_group[0], dataset, tokenizer)
        g_dict.update(res)
    else:
        pool = multiprocessing.Pool(processes=args.n_processes)
        worker_results = []
        for pid in range(args.n_processes):
            worker_results.append(pool.apply_async(worker_annotate, args=(
                pid, args, generate_eids_group[pid], dataset, tokenizer
            )))
        for r in worker_results:
            worker_g_dict = r.get()
            g_dict.update(worker_g_dict)
        pool.close()
        pool.join()

    # Recompute overall metrics across all workers
    em_and_f1 = TaTQAEmAndF1()
    for eid, item in g_dict.items():
        ori = item['ori_data_item']
        gold_answer = ori['answer_text']
        if len(gold_answer) == 2 and gold_answer[-1] == '###':
            gold_answer = gold_answer[0]
        ground_truth = {
            'answer': gold_answer,
            'answer_type': ori['answer_type'],
            'scale': ori['scale'],
            'answer_from': 'table-text',
        }
        em_and_f1(ground_truth=ground_truth, prediction=item['pred_answer'], pred_scale='')

    overall_em, overall_f1, scale_score, _ = em_and_f1.get_overall_metric()
    num_failed = sum([1 for i in g_dict.values() if len(i['generations']) == 0])
    print(f"\nElapsed time: {time.time() - start_time:.1f}s")
    print(f"Total examples: {len(g_dict)}")
    print(f"Failed generations: {num_failed}")
    print(f"Overall EM: {overall_em:.4f}")
    print(f"Overall F1: {overall_f1:.4f}")
    print(f"Scale Accuracy: {scale_score:.4f}")

    with open(os.path.join(args.save_dir, args.output_program_file), 'w') as f:
        json.dump(g_dict, f, indent=4)


if __name__ == '__main__':
    if platform.system() == "Darwin":
        multiprocessing.set_start_method('spawn')

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='tatqa')
    parser.add_argument('--dataset_split', type=str, default='validation')
    parser.add_argument('--api_config_file', type=str, default='key.txt')
    parser.add_argument('--output_program_file', type=str, default='pot_tatqa_validation.json')
    parser.add_argument('--save_dir', type=str, default='results/')
    parser.add_argument('--engine', type=str, default='Qwen/Qwen2.5-3B-Instruct')
    parser.add_argument('--n_processes', type=int, default=1)
    parser.add_argument('--max_generation_tokens', type=int, default=256)
    parser.add_argument('--max_api_total_tokens', type=int, default=4001)
    parser.add_argument('--max_sample_num', type=int, default=99999999)
    parser.add_argument('--sampling_n', type=int, default=1)
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('--debug', action='store_true')

    args = parser.parse_args()

    print("POT TatQA Args:")
    for k in args.__dict__:
        print(f"  {k}: {args.__dict__[k]}")

    main()
