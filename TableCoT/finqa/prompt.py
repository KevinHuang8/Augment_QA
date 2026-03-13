import json
import argparse
import tqdm
import re
import os
from datetime import datetime
from openai import OpenAI

parser = argparse.ArgumentParser()
parser.add_argument("--option", default='cot', type=str)
parser.add_argument("--model", default='Qwen/Qwen2.5-3B-Instruct', type=str)
parser.add_argument("--start", required=True, type=int)
parser.add_argument("--end", required=True, type=int)
parser.add_argument("--dry_run", default=False, action="store_true",
    help="whether it's a dry run or real run.")
parser.add_argument("--temperature", type=float, default=0.0)
parser.add_argument("--base_url", type=str, default="http://localhost:8000/v1")

demonstration = {}

demonstration['direct'] = """
Read the table and context below to answer the following question.

Context: future minimum lease payments for all non-cancelable operating leases at may 31 , 2013 were as follows :
2014 | $11,057
2015 | 8,985
2016 | 7,378
2017 | 6,700
2018 | 6,164
Thereafter | 16,812
Total future minimum lease payments | $57,096

Question: what percentage of lease payments will be paid out in the first year?
The answer is 0.19366.

Context: balance of unrecognized tax positions.
Balance at January 1, 2013 | $180,993
Increases in current period tax positions | 27,229
Decreases in prior period measurement of tax positions | (30,275)
Balance at December 31, 2013 | $177,947
Increases in current period tax positions | 53,818
Decreases in prior period measurement of tax positions | (36,528)
Balance at December 31, 2014 | $195,237

Question: what was the net change in tax positions in 2014?
The answer is 17447.0.

Context: office locations and approximate square footage.
Location | Approximate Square Footage
Alpharetta, Georgia | 254,000
Jersey City, New Jersey | 107,000
Arlington, Virginia | 102,000

Context: as of december 31 , 2012 , 165000 square feet in alpharetta georgia was not yet leased .
Question: as of december 2012 what is the percent of the square footage not leased to the total square footage in alpharetta , georgia?
The answer is 0.64961.
"""

demonstration['cot'] = """
Read the table and context below to answer the following question.

Context: future minimum lease payments for all non-cancelable operating leases at may 31 , 2013 were as follows :
2014 | $11,057
2015 | 8,985
2016 | 7,378
2017 | 6,700
2018 | 6,164
Thereafter | 16,812
Total future minimum lease payments | $57,096

Question: what percentage of lease payments will be paid out in the first year?
Explanation: The first year payment (2014) is $11,057. The total is $57,096. The percentage is 11057 / 57096 = 0.1937, or about 19.4%. Therefore, the answer is 0.19366.

Context: balance of unrecognized tax positions.
Balance at January 1, 2013 | $180,993
Increases in current period tax positions | 27,229
Decreases in prior period measurement of tax positions | (30,275)
Balance at December 31, 2013 | $177,947
Increases in current period tax positions | 53,818
Decreases in prior period measurement of tax positions | (36,528)
Balance at December 31, 2014 | $195,237

Question: what was the net change in tax positions in 2014?
Explanation: In 2014, increases were 53,818 and decreases were -36,528. Net change = 53818 + (-36528) = 17290. Adding penalties and interest of 157 gives 17290 + 157 = 17447. Therefore, the answer is 17447.0.

Context: office locations and approximate square footage.
Location | Approximate Square Footage
Alpharetta, Georgia | 254,000
Jersey City, New Jersey | 107,000
Arlington, Virginia | 102,000

Context: as of december 31 , 2012 , 165000 square feet in alpharetta georgia was not yet leased .
Question: as of december 2012 what is the percent of the square footage not leased to the total square footage in alpharetta , georgia?
Explanation: The not-leased square footage is 165,000. The total for Alpharetta is 254,000. The percentage is 165000 / 254000 = 0.6496, about 64.9%. Therefore, the answer is 0.64961.
"""


def table_to_str(table_ori):
    rows = []
    for row in table_ori:
        rows.append(" | ".join(str(c).strip() for c in row))
    return "\n".join(rows)


if __name__ == "__main__":
    args = parser.parse_args()

    client = OpenAI(base_url=args.base_url, api_key="dummy")

    data_path = os.path.join(os.path.dirname(__file__), "finqa_test.json")
    with open(data_path) as f:
        finqa = json.load(f)

    now = datetime.now()
    dt_string = now.strftime("%d_%H_%M")

    entries = finqa[args.start:args.end]

    if not args.dry_run:
        model_tag = args.model.split('/')[-1]
        fw = open(f'outputs/response_s{args.start}_e{args.end}_{args.option}_{model_tag}_{dt_string}.json', 'w')
        fw.write(json.dumps({'demonstration': demonstration[args.option]}) + '\n')

    for i, entry in enumerate(tqdm.tqdm(entries)):
        idx = args.start + i
        question = entry['qa']['question']
        answer = str(entry['qa']['exe_ans'])
        table_str = table_to_str(entry.get('table_ori', entry.get('table', [])))
        pre_text = " ".join(entry.get('pre_text', []))
        post_text = " ".join(entry.get('post_text', []))

        prompt = demonstration[args.option] + '\n'
        if pre_text:
            prompt += f'Context: {pre_text}\n'
        prompt += table_str + '\n'
        if post_text:
            prompt += f'Context: {post_text}\n'
        prompt += '\nQuestion: ' + question
        prompt += '\nCompute the exact numeric answer step by step, then end with "Therefore, the answer is [number]."\nExplanation:'

        if args.dry_run:
            print(prompt)
            print('answer: ', answer)
        else:
            response = client.chat.completions.create(
                model=args.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=args.temperature,
                max_tokens=256,
                top_p=1,
            )
            response = response.choices[0].message.content.strip()

            tmp = {'idx': idx, 'id': entry.get('id', str(idx)),
                   'question': question, 'response': response, 'answer': answer}
            fw.write(json.dumps(tmp) + '\n')

    if not args.dry_run:
        fw.close()
