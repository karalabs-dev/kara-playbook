import re, statistics
from collections import Counter


def score_answer(predicted, ground_truth, answer_type):
    """Score a single answer against ground truth. Returns 0.0-1.0."""
    if predicted is None:
        return 0.0
    predicted_str = str(predicted).strip()
    try:
        if answer_type == 'bool':
            lower = predicted_str.lower()
            if any(w in lower for w in ('false', 'no ', 'not ', 'does not', "doesn\'t", 'none')):
                pred_bool = False
            elif any(w in lower for w in ('true', 'yes', 'it does', 'it has')):
                pred_bool = True
            else:
                pred_bool = lower in ('true', 'yes', '1')
            return 1.0 if pred_bool == ground_truth else 0.0

        elif answer_type == 'int':
            nums = re.findall(r'\b(\d+)\b', predicted_str)
            if not nums:
                return 0.0
            for n in nums:
                if abs(int(n) - ground_truth) / max(ground_truth, 1) <= 0.1:
                    return 1.0
            return 0.0

        elif answer_type == 'str':
            return 1.0 if ground_truth.lower() in predicted_str.lower() else 0.0

        elif answer_type == 'list':
            if isinstance(predicted, str):
                predicted_str = predicted_str.strip('[] ')
                items = re.split(r'[,\n]|\s*[-\u2022\*]\s*', predicted_str)
                predicted = [s.strip().strip("\'\"\\\'`") for s in items if s.strip()]
            pred_set = set(s.lower().removesuffix('.py').replace('src/click/', '').strip() for s in predicted if s.strip())
            truth_set = set(s.lower().removesuffix('.py').replace('src/click/', '').strip() for s in ground_truth if s.strip())
            hits = len(pred_set & truth_set)
            precision = hits / max(len(pred_set), 1)
            recall = hits / max(len(truth_set), 1)
            if precision + recall == 0:
                return 0.0
            return 2 * precision * recall / (precision + recall)
    except (ValueError, TypeError):
        return 0.0
    return 0.0


def parse_answers(text):
    """Extract Q1: [confidence] answer from agent output."""
    answers = {}
    confidences = {}
    text = re.sub(r'\*\*Q?(\d+)[.:\)]?\*\*[:\s]*', r'Q\1: ', text)
    text = re.sub(r'^(\d+)[.)\]]\s+', r'Q\1: ', text, flags=re.MULTILINE)

    for m in re.finditer(r'Q(\d+):\s*(.+?)(?=\nQ\d+:|\n\d+[.)\]]|$)', text, re.DOTALL):
        qid = int(m.group(1))
        raw = m.group(2).strip()
        conf_match = re.match(r'\[([0-9.]+%?)\]\s*(.*)', raw, re.DOTALL)
        if conf_match:
            try:
                conf_str = conf_match.group(1)
                if conf_str.endswith('%'):
                    confidences[qid] = float(conf_str[:-1]) / 100
                else:
                    confidences[qid] = float(conf_str)
            except ValueError:
                confidences[qid] = None
            answers[qid] = conf_match.group(2).strip()
        else:
            answers[qid] = raw
            confidences[qid] = None
    return answers, confidences


def score_run(answers, ground_truth):
    """Score all answers against ground truth."""
    scores = {}
    for gt in ground_truth:
        qid = gt['id']
        predicted = answers.get(qid)
        score = score_answer(predicted, gt['a'], gt['type'])
        scores[qid] = {
            'score': score,
            'cat': gt['cat'],
            'predicted': predicted,
            'truth': gt['a'],
            'type': gt['type'],
        }
    return scores


# pricing: gemini-3.1-flash-lite-preview
INPUT_COST = 0.075 / 1_000_000
OUTPUT_COST = 0.30 / 1_000_000

# pricing: gemini-3-flash-preview (thinking tokens = output rate)
FLASH_INPUT_COST = 0.15 / 1_000_000
FLASH_OUTPUT_COST = 0.60 / 1_000_000
