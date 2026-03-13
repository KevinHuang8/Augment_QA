"""
Compute exact-match accuracy for TATQA TableCoT predictions.

Usage:
    python compute_score.py --inputs "outputs/response_*.json"
"""

import json
import glob
import argparse
import re

parser = argparse.ArgumentParser()
parser.add_argument("--inputs", required=True, type=str)
parser.add_argument("--tolerance", type=float, default=2e-3)

SCALE_FACTORS = {'thousand': 1e3, 'million': 1e6, 'billion': 1e9, 'percent': 1}


def extract_answer(response: str) -> str:
    m = re.search(r"[Tt]he answer is ([^\.\n]+)", response)
    if m:
        return m.group(1).strip().rstrip(".")
    numbers = re.findall(r"-?[\d,]+(?:\.\d+)?%?", response)
    if numbers:
        return numbers[-1].replace(",", "")
    return response.split('\n')[0].strip()


def normalise(text: str):
    text = text.strip()
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


def exact_match(pred: str, gold_list, scale: str, tol: float) -> bool:
    # Normalise gold to always be a list
    if not isinstance(gold_list, list):
        gold_list = [gold_list]
    # Numeric match
    p = normalise(pred)
    scale_factor = SCALE_FACTORS.get(scale, 1)
    for g_str in gold_list:
        g = normalise(str(g_str))
        if p is not None and g is not None:
            # Try direct comparison
            if g == 0 and abs(p) < tol:
                return True
            if g != 0 and abs(p - g) / abs(g) < tol:
                return True
            # Try scaled comparison (gold in millions, pred not scaled)
            if scale_factor != 1 and g != 0:
                g_scaled = g * scale_factor
                if abs(p - g_scaled) / abs(g_scaled) < tol:
                    return True
        # String match
        if pred.strip().lower() == str(g_str).strip().lower():
            return True
    return False


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
                gold = record["answer"]  # list of strings
                scale = record.get("scale", "")

                if exact_match(pred, gold, scale, args.tolerance):
                    correct += 1
                total += 1

        if total == 0:
            print("  No predictions found.")
        else:
            print(f"  Exact match: {correct/total:.4f}  ({correct}/{total})")
