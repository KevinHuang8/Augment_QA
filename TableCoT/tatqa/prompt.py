import os
import json
import argparse
import tqdm
from datetime import datetime
from openai import OpenAI

parser = argparse.ArgumentParser()
parser.add_argument("--option", default='cot', type=str)
parser.add_argument("--model", default='Qwen/Qwen2.5-3B-Instruct', type=str)
parser.add_argument("--start", required=True, type=int)
parser.add_argument("--end", required=True, type=int)
parser.add_argument("--dry_run", default=False, action="store_true")
parser.add_argument("--temperature", type=float, default=0.0)
parser.add_argument("--base_url", type=str, default="http://localhost:8000/v1")

demonstration = {}

demonstration['direct'] = """
Read the table and context below to answer the following question.

Context: Net sales by segment were as follows:
 | 2019 | 2018 | 2017
Fixed Price | $ 1,452.4 | $ 1,146.2 | $ 1,036.9
Other | 44.1 | 56.7 | 70.8

Question: What is the change in Other in 2019 from 2018?
The answer is -12.6.

Context: Net sales by segment were as follows:
 | 2019 | 2018 | 2017
Fixed Price | $ 1,452.4 | $ 1,146.2 | $ 1,036.9
Other | 44.1 | 56.7 | 70.8

Question: What is the percentage change in Other in 2019 from 2018?
The answer is -22.22.
"""

demonstration['cot'] = """
Read the table and context below to answer the following question.

Context: Net sales by segment were as follows:
 | 2019 | 2018 | 2017
Fixed Price | $ 1,452.4 | $ 1,146.2 | $ 1,036.9
Other | 44.1 | 56.7 | 70.8

Question: What is the change in Other in 2019 from 2018?
Explanation: Other in 2019 is 44.1 and in 2018 is 56.7. The change is 44.1 - 56.7 = -12.6 million. Therefore, the answer is -12.6.

Context: Net sales by segment were as follows:
 | 2019 | 2018 | 2017
Fixed Price | $ 1,452.4 | $ 1,146.2 | $ 1,036.9
Other | 44.1 | 56.7 | 70.8

Question: What is the percentage change in Other in 2019 from 2018?
Explanation: Other changed from 56.7 in 2018 to 44.1 in 2019. Percentage change = (44.1 - 56.7) / 56.7 * 100 = -12.6 / 56.7 * 100 = -22.22%. Therefore, the answer is -22.22.
"""


def table_to_str(table_ori):
    rows = []
    for row in table_ori:
        rows.append(" | ".join(str(c).strip() for c in row))
    return "\n".join(rows)


if __name__ == "__main__":
    args = parser.parse_args()

    client = OpenAI(base_url=args.base_url, api_key="dummy")

    data_path = os.path.join(os.path.dirname(__file__),
                             "../../datasets/tatqa_validation.json")
    with open(data_path) as f:
        tatqa = json.load(f)

    # Flatten: each doc has multiple questions
    flat = []
    for doc in tatqa:
        table_str = table_to_str(doc['table']['table'])
        para_text = " ".join(p['text'] for p in doc['paragraphs'])
        for q in doc['questions']:
            flat.append({
                'question': q['question'],
                'answer': q['answer'],
                'scale': q.get('scale', ''),
                'answer_type': q.get('answer_type', ''),
                'table_str': table_str,
                'para_text': para_text,
                'uid': q['uid'],
            })

    entries = flat[args.start:args.end]
    print(f"Running on {len(entries)} questions.")

    now = datetime.now()
    dt_string = now.strftime("%d_%H_%M")
    model_tag = args.model.split('/')[-1]

    if not args.dry_run:
        fw = open(f'outputs/response_s{args.start}_e{args.end}_{args.option}_{model_tag}_{dt_string}.json', 'w')
        fw.write(json.dumps({'demonstration': demonstration[args.option]}) + '\n')

    for entry in tqdm.tqdm(entries):
        question = entry['question']
        answer = entry['answer']

        prompt = demonstration[args.option] + '\n'
        prompt += f'Context: {entry["para_text"][:400]}\n'
        prompt += entry['table_str'] + '\n\n'
        prompt += 'Question: ' + question + '\n'
        prompt += 'Compute the exact answer step by step, then end with "Therefore, the answer is [value]."\nExplanation:'

        if args.dry_run:
            print(prompt)
            print('answer:', answer)
            continue

        response = client.chat.completions.create(
            model=args.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=args.temperature,
            max_tokens=256,
            top_p=1,
        )
        response = response.choices[0].message.content.strip()

        tmp = {'uid': entry['uid'], 'question': question, 'response': response,
               'answer': answer, 'scale': entry['scale'], 'answer_type': entry['answer_type']}
        fw.write(json.dumps(tmp) + '\n')

    if not args.dry_run:
        fw.close()
        print("Done.")
