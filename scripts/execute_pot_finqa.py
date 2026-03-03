"""
POT (Program of Thought) baseline for FinQA dataset.
Adapted from execute_pot.py (which only supports WikiTQ).
"""

import time
import json
import argparse
import copy
import os
import random
import pyrootutils
pyrootutils.setup_root(".project-root", pythonpath=True)

import func_timeout
from typing import List
import platform
import multiprocessing
from tqdm import tqdm
from transformers import AutoTokenizer
from tenacity import retry, stop_after_attempt, wait_exponential
from openai import OpenAI

from utils.evaluator import Evaluator
from generation.generator import Generator
from utils.utils import load_data_split

ROOT_DIR = os.path.join(os.path.dirname(__file__), "../")

VLLM_BASE_URL = "http://localhost:8000/v1"


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


def linearize_finqa_table(data_item, n_rows=1000):
    """Linearize a FinQA table with report context (pre_text + post_text)."""
    lines = []

    # Add report context
    report_text = []
    if 'pre_text' in data_item and data_item['pre_text']:
        report_text.extend(data_item['pre_text'])
    if 'post_text' in data_item and data_item['post_text']:
        report_text.extend(data_item['post_text'])
    if report_text:
        lines.append("Report:")
        lines.append(' '.join(report_text))
        lines.append("")

    # Add table
    lines.append("Table:")
    table = data_item['table']
    lines.append(' | '.join(table['header']).strip())
    for row in table['rows'][:n_rows]:
        lines.append(' | '.join(row).strip())

    # Add question
    lines.append(f"\nQ: {data_item['question']}\n")
    return '\n'.join(lines)


# Few-shot examples for FinQA POT
FINQA_POT_EXAMPLES = """Read the following report and table, then write a Python program to answer the question.

Report:
contingent acquisition obligations the following table details the estimated future contingent acquisition obligations payable in cash as of december 31 , 2009 .
Table:
filledcolumnname | 2010 | 2011 | 2012 | 2013 | 2014 | thereafter | total
deferred acquisition payments | 20.5 | 34.8 | 1.2 | 1.1 | 2.1 | 0.3 | 60.0
redeemable noncontrolling interests and call options with affiliates | 44.4 | 47.9 | 40.5 | 36.3 | 3.3 | 0.0 | 172.4
total contingent acquisition payments | 64.9 | 82.7 | 41.7 | 37.4 | 5.4 | 0.3 | 232.4

Q: what percentage decrease occurred from 2011-2012 for deferred acquisition payments?
deferred_2011 = 34.8
deferred_2012 = 1.2
ans = (deferred_2011 - deferred_2012) / deferred_2011 * 100


Read the following report and table, then write a Python program to answer the question.

Report:
management's financial discussion and analysis net revenue 2008 compared to 2007 net revenue consists of operating revenues net of fuel and purchased power expenses.
Table:
filledcolumnname | amount ( in millions )
2007 net revenue | 442.3
volume/weather | 4.6
reserve equalization | 3.3
securitization transition charge | 9.1
fuel recovery | 7.5
other | 10.1
2008 net revenue | 440.9

Q: what is the percent change in net revenue between 2007 and 2008?
net_revenue_2007 = 442.3
net_revenue_2008 = 440.9
ans = (net_revenue_2008 - net_revenue_2007) / net_revenue_2007 * 100


Read the following report and table, then write a Python program to answer the question.

Report:
purchases of equity securities during 2014 , we repurchased 33035204 shares of our common stock at an average price of $ 100.24 . effective january 1 , 2014 , our board of directors authorized the repurchase of up to 120 million shares by december 31 , 2017 .
Table:
period | total number of shares purchased | average price paid per share | total number of shares purchased as part of a publicly announced plan or program | maximum number of shares that may yet be purchased under the plan or program
oct . 1 through oct . 31 | 3087549 | 107.59 | 3075000 | 92618000
nov . 1 through nov . 30 | 1877330 | 119.84 | 1875000 | 90743000
dec . 1 through dec . 31 | 2787108 | 116.54 | 2786400 | 87956600
total | 7751987 | 113.77 | 7736400 | n/a

Q: what was the total number of shares purchased in october and november?
shares_oct = 3087549
shares_nov = 1877330
ans = shares_oct + shares_nov


Read the following report and table, then write a Python program to answer the question.

Report:
the company had mass layoffs in 2008 with severance costs totaling 14.6 million. restructuring costs were reduced in 2009 to 3.2 million as operations stabilized.
Table:
filledcolumnname | 2009 | 2008
severance costs | 3.2 | 14.6
facility closure costs | 1.5 | 4.8
total restructuring costs | 4.7 | 19.4

Q: what is the ratio of severance costs in 2008 to 2009?
severance_2008 = 14.6
severance_2009 = 3.2
ans = severance_2008 / severance_2009"""


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

    g_dict = dict()
    total_num, correct_num = 0, 0

    pbar = tqdm(g_eids, desc=f"POT-FinQA Worker {pid}", position=pid, leave=True)
    for idx, g_eid in enumerate(pbar):
        g_data_item = dataset[g_eid]
        g_dict[g_eid] = {
            'generations': [],
            'ori_data_item': copy.deepcopy(g_data_item)
        }

        few_shot_prompt = FINQA_POT_EXAMPLES

        generate_prompt = '\nRead the following report and table, then write a Python program to answer the question:\n'
        max_row = 100
        query_table = linearize_finqa_table(g_data_item, n_rows=max_row)
        max_prompt_tokens = args.max_api_total_tokens - args.max_generation_tokens
        while len(tokenizer.tokenize(query_table)) >= max_prompt_tokens - 2000:
            max_row -= 10
            query_table = linearize_finqa_table(g_data_item, n_rows=max_row)
        generate_prompt += query_table
        prompt = few_shot_prompt + "\n\n" + generate_prompt

        # Shrink few-shot if prompt is too long (remove examples from the front)
        prompt_text = prompt
        while len(tokenizer.tokenize(prompt_text)) >= max_prompt_tokens:
            parts = few_shot_prompt.split("\n\n\nRead the following")
            if len(parts) <= 1:
                break
            few_shot_prompt = "Read the following" + "\n\n\nRead the following".join(parts[1:])
            prompt = few_shot_prompt + "\n\n" + generate_prompt
            prompt_text = prompt

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

        # Evaluate
        g_dict[g_eid]['pred_answer'] = ans
        gold_answer = g_data_item['answer_text']
        score = Evaluator().evaluate(
            ans,
            gold_answer,
            dataset='finqa',
            question=g_data_item['question']
        )
        g_dict[g_eid]['score'] = score
        g_dict[g_eid]['error_msg'] = error_msg
        if score == 1:
            correct_num += 1
        total_num += 1
        pbar.set_postfix(acc=f"{correct_num}/{total_num} ({correct_num / total_num:.2%})")

    return g_dict


def main():
    # Build paths
    args.api_config_file = os.path.join(ROOT_DIR, args.api_config_file)
    args.save_dir = os.path.join(ROOT_DIR, args.save_dir)
    os.makedirs(args.save_dir, exist_ok=True)

    # Load dataset
    start_time = time.time()
    dataset = load_data_split(args.dataset, args.dataset_split)
    dataset = dataset.select(range(min(args.max_sample_num, len(dataset))))

    generate_eids = list(range(len(dataset)))
    generate_eids_group = [[] for _ in range(args.n_processes)]
    for g_eid in generate_eids:
        generate_eids_group[(int(g_eid) + random.randrange(args.n_processes)) % args.n_processes].append(g_eid)

    print(f'\n******* Running POT on FinQA ({len(dataset)} examples) *******')
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=args.engine)

    g_dict = dict()
    worker_results = []
    if args.n_processes == 1:
        res = worker_annotate(0, args, generate_eids_group[0], dataset, tokenizer)
        g_dict.update(res)
    else:
        pool = multiprocessing.Pool(processes=args.n_processes)
        for pid in range(args.n_processes):
            worker_results.append(pool.apply_async(worker_annotate, args=(
                pid, args, generate_eids_group[pid], dataset, tokenizer
            )))
        for r in worker_results:
            worker_g_dict = r.get()
            g_dict.update(worker_g_dict)
        pool.close()
        pool.join()

    num_failed = sum([1 for i in g_dict.values() if len(i['generations']) == 0])
    n_correct_samples = sum([item['score'] for item in g_dict.values()])
    print(f"\nElapsed time: {time.time() - start_time:.1f}s")
    print(f"Total examples: {len(g_dict)}")
    print(f"Failed generations: {num_failed}")
    print(f'Overall Accuracy: {n_correct_samples}/{len(g_dict)} = {n_correct_samples / len(g_dict):.4f}')

    # Save results
    with open(os.path.join(args.save_dir, args.output_program_file), 'w') as f:
        json.dump(g_dict, f, indent=4)


if __name__ == '__main__':
    if platform.system() == "Darwin":
        multiprocessing.set_start_method('spawn')

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='finqa')
    parser.add_argument('--dataset_split', type=str, default='test')
    parser.add_argument('--api_config_file', type=str, default='key.txt')
    parser.add_argument('--output_program_file', type=str, default='pot_finqa_test.json')
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

    print("POT FinQA Args:")
    for k in args.__dict__:
        print(f"  {k}: {args.__dict__[k]}")

    main()