"""
Compute exact-match accuracy for FinQA TableCoT outputs.

Usage:
    python compute_score.py --inputs "outputs/response_*.json"
"""

import json
import glob
import argparse
import re

parser = argparse.ArgumentParser()
parser.add_argument("--inputs", required=True, type=str)
parser.add_argument("--tolerance", type=float, default=2e-3,
    help="Relative tolerance for numeric comparison (default 0.2%%).")


def extract_answer(response: str) -> str:
    """Extract the final answer from the model's CoT response."""
    # Try "Therefore, the answer is X." or "the answer is X"
    m = re.search(r"[Tt]he answer is ([^\.\n]+)", response)
    if m:
        return m.group(1).strip().rstrip(".")
    # Fallback: last number in the response
    numbers = re.findall(r"-?[\d,]+(?:\.\d+)?", response)
    if numbers:
        return numbers[-1].replace(",", "")
    return response.split('\n')[0].strip()


def normalise(text: str):
    text = text.strip()
    list_match = re.match(r"^\[.?'(.+?)'.?\]$", text)
    if list_match:
        text = list_match.group(1).strip()
    is_percent = "%" in text
    paren_match = re.match(r"^\(([0-9,\.]+)\)$", text.replace("$", "").replace(" ", ""))
    if paren_match:
        text = "-" + paren_match.group(1)
    text = text.replace(",", "").replace("%", "").replace("$", "").strip()
    try:
        val = float(text)
        if is_percent:
            val = val / 100
        return val
    except ValueError:
        return None


def exact_match(pred: str, gold: str, tol: float) -> bool:
    p, g = normalise(pred), normalise(gold)
    if p is not None and g is not None:
        if g == 0:
            return abs(p - g) < tol
        return abs(p - g) / abs(g) < tol
    return pred.strip().lower() == gold.strip().lower()


if __name__ == "__main__":
    args = parser.parse_args()

    for filename in glob.glob(args.inputs):
        correct = 0
        total = 0
        print(f"\nEvaluating: {filename}")

        with open(filename) as f:
            for line in f:
                record = json.loads(line.strip())
                if "response" not in record:
                    continue

                pred = extract_answer(record["response"])
                gold = str(record["answer"]).strip()

                if exact_match(pred, gold, args.tolerance):
                    correct += 1
                total += 1

        if total == 0:
            print("  No predictions found.")
        else:
            print(f"  Exact match: {correct/total:.4f}  ({correct}/{total})")
