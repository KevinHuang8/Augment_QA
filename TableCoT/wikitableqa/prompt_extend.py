import os
import json
import random
import argparse
import tqdm
import sys
from datetime import datetime
from openai import OpenAI

parser = argparse.ArgumentParser()
parser.add_argument("--option", default='cot', type=str)
parser.add_argument("--model", default='text-davinci-002', type=str)
parser.add_argument("--start", required=True, type=int)
parser.add_argument("--end", required=True, type=int)
parser.add_argument("--dry_run", default=False, action="store_true",
    help="whether it's a dry run or real run.")
parser.add_argument("--temperature", type=float, default=0.0)
parser.add_argument("--shots", type=int, default=4, choices=[4, 8],
    help="Number of in-context demonstration examples (4 or 8).")

TABLE_HEADER = """Read the table below regarding "2008 Clásica de San Sebastián" to answer the following questions.

Rank | Cyclist | Team | Time | UCI ProTour Points
1 | Alejandro Valverde (ESP) | Caisse d'Epargne | 5h 29' 10 | 40
2 | Alexandr Kolobnev (RUS) | Team CSC Saxo Bank | s.t. | 30
3 | Davide Rebellin (ITA) | Gerolsteiner | s.t. | 25
4 | Paolo Bettini (ITA) | Quick Step | s.t. | 20
5 | Franco Pellizotti (ITA) | Liquigas | s.t. | 15
6 | Denis Menchov (RUS) | Rabobank | s.t. | 11
7 | Samuel Sánchez (ESP) | Euskaltel-Euskadi | s.t. | 7
8 | Stéphane Goubert (FRA) | Ag2r-La Mondiale | + 2 | 5
9 | Haimar Zubeldia (ESP) | Euskaltel-Euskadi | + 2 | 3
10 | David Moncoutié (FRA) | Cofidis | + 2 | 1
"""

DIRECT_4 = """
Question: which country had the most cyclists finish within the top 10?
The answer is Italy.

Question: how many players got less than 10 points?
The answer is 4.

Question: how many points does the player from rank 3, rank 4 and rank 5 combine to have?
The answer is 60.

Question: who spent the most time in the 2008 Clásica de San Sebastián.
The answer is David Moncoutié.
"""

DIRECT_EXTRA = """
Question: what is the total number of UCI ProTour Points awarded?
The answer is 157.

Question: which team had two cyclists finish in the top 10?
The answer is Euskaltel-Euskadi.

Question: how many cyclists finished with the same time as the winner?
The answer is 6.

Question: what is the difference in points between 1st place and 2nd place?
The answer is 10.
"""

COT_4 = """
Question: which country had the most cyclists finish within the top 10?
Explanation: ITA occurs three times in the table, more than any others. Therefore, the answer is Italy.

Question: how many players got less than 10 points?
Explanation: Samuel Sánchez,  Stéphane Goubert, Haimar Zubeldia and David Moncoutié received less than 10 points.  Therefore, the answer is 4.

Question: how many points does the player from rank 3, rank 4 and rank 5 combine to have?
Explanation: rank 3 has 25 points, rank 4 has 20 points, rank 5 has 15 points, they combine to have a total of 60 points. Therefore, the answer is 60.

Question: who spent the most time in the 2008 Clásica de San Sebastián?
Explanation: David Moncoutié spent the most time to finish the game and ranked the last. Therefore, the answer is David Moncoutié.
"""

COT_EXTRA = """
Question: what is the total number of UCI ProTour Points awarded?
Explanation: The points awarded are 40+30+25+20+15+11+7+5+3+1=157. Therefore, the answer is 157.

Question: which team had two cyclists finish in the top 10?
Explanation: Samuel Sánchez (rank 7) and Haimar Zubeldia (rank 9) are both from Euskaltel-Euskadi, making it the only team with two top-10 finishers. Therefore, the answer is Euskaltel-Euskadi.

Question: how many cyclists finished with the same time as the winner?
Explanation: Cyclists ranked 2 through 7 all have "s.t." (same time), so 6 cyclists finished with the same time as the winner. Therefore, the answer is 6.

Question: what is the difference in points between 1st place and 2nd place?
Explanation: 1st place has 40 points and 2nd place has 30 points, so the difference is 40-30=10. Therefore, the answer is 10.
"""

def build_demonstration(option, shots):
    header = "read the question first, and then answer the given question. \n" if option == 'direct' else ""
    base = DIRECT_4 if option == 'direct' else COT_4
    extra = DIRECT_EXTRA if option == 'direct' else COT_EXTRA
    demo = "\n" + TABLE_HEADER + "\n" + header + base
    if shots == 8:
        demo += extra
    return demo

demonstration = {
    'direct': {4: build_demonstration('direct', 4), 8: build_demonstration('direct', 8)},
    'cot':    {4: build_demonstration('cot', 4),    8: build_demonstration('cot', 8)},
}

if __name__ == "__main__":
    args = parser.parse_args()

    client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")

    with open(f'test_qa.json') as f:
        wikitableqa = json.load(f)

    now = datetime.now() 
    dt_string = now.strftime("%d_%H_%M")

    keys = list(wikitableqa.keys())[args.start:args.end]

    correct = 0
    wrong = 0

    if not args.dry_run:
        model_version = args.model.split('/')[-1]
        fw = open(f'outputs/response_s{args.start}_e{args.end}_{args.option}_{model_version}_{dt_string}.json', 'w')
        tmp = {'demonstration': demonstration[args.option][args.shots]}
        fw.write(json.dumps(tmp) + '\n')

    for key in tqdm.tqdm(keys):
        entry = wikitableqa[key]

        question = entry['question']
        answer = entry['answer']

        #### Formalizing the k-shot demonstration. #####
        prompt = demonstration[args.option][args.shots] + '\n'
        prompt += f'Read the table blow regarding "{entry["title"]}" to answer the following question.\n\n'
        if 'davinci' not in args.model:
            prompt += '\n'.join(entry['table'].split('\n')[:15])
        else:
            prompt += entry['table'] + '\n'
        prompt += 'Question: ' + question + '\nExplanation:'

        if args.dry_run:
            print(prompt)
            print('answer: ', answer)
        else:
            response = client.chat.completions.create(
              model=args.model,
              messages=[{"role": "user", "content": prompt}],
              temperature=0.0,
              max_tokens=256,
              top_p=1,
            )

            response = response.choices[0].message.content.strip().split('\n')[0]

            tmp = {'key': key, 'question': question, 'response': response, 'answer': answer, 'table_id': entry['table_id']}

            fw.write(json.dumps(tmp) + '\n')

    if not args.dry_run:
        print(correct, wrong, correct / (correct + wrong + 0.001))
        fw.close()
