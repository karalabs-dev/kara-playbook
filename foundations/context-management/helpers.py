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
            # normalize markdown escapes (e.g. web\_lookup -> web_lookup)
            normalized = predicted_str.lower().replace('\\_', '_')
            return 1.0 if ground_truth.lower() in normalized else 0.0

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


# pricing: gemini-2.5-flash-lite
INPUT_COST = 0.10 / 1_000_000
OUTPUT_COST = 0.40 / 1_000_000

# pricing: gemini-3-flash-preview (thinking tokens = output rate)
FLASH_INPUT_COST = 0.15 / 1_000_000
FLASH_OUTPUT_COST = 0.60 / 1_000_000


# ── Chart functions ──────────────────────────────────────────────────

import numpy as np
import matplotlib.pyplot as plt

COLORS = {"baseline": "red", "sliding_window": "orange", "running_summary": "green", "scratchpad": "blue"}
LABELS = {"baseline": "Baseline", "sliding_window": "Sliding Window", "running_summary": "Running Summary", "scratchpad": "Scratchpad"}


def _snap_tokens(turn):
    snap = turn.get("snapshot") or {}
    return snap.get("total_tokens_in_context", 0) or turn.get("input_tokens", 0)


def plot_baseline_rot(baseline_runs, n_turns, token_window, trigger_threshold, save_path="results/baseline_rot.png"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for run in baseline_runs:
        ax1.plot(range(1, n_turns+1), [_snap_tokens(t) for t in run["turns"]], alpha=0.4, color="red")
    avg_tokens = [statistics.mean([_snap_tokens(r["turns"][i]) for r in baseline_runs]) for i in range(n_turns)]
    ax1.plot(range(1, n_turns+1), avg_tokens, color="red", linewidth=2, label="avg")
    ax1.axhline(y=token_window, color="black", linestyle="--", alpha=0.5, label=f"TOKEN_WINDOW ({token_window})")
    ax1.axhline(y=token_window * trigger_threshold, color="gray", linestyle=":", alpha=0.5, label=f"Trigger ({trigger_threshold:.0%})")
    ax1.set(xlabel="Turn", ylabel="Context Tokens (approx)", title="Context Growth (baseline)")
    ax1.legend(); ax1.grid(True, alpha=0.3)

    for run in baseline_runs:
        ax2.plot(range(1, n_turns+1), [t["score"] for t in run["turns"]], alpha=0.4, color="red")
    avg_scores = [statistics.mean([r["turns"][i]["score"] for r in baseline_runs]) for i in range(n_turns)]
    ax2.plot(range(1, n_turns+1), avg_scores, color="red", linewidth=2, label="avg")
    ax2.set(xlabel="Turn", ylabel="Accuracy", title="Accuracy (baseline)", ylim=(-0.05, 1.05))
    ax2.legend(); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_token_growth(strategies, n_turns, token_window, trigger_threshold, save_path="results/token_growth.png"):
    fig, ax = plt.subplots(figsize=(12, 6))
    for name, runs in strategies.items():
        per_turn = [[_snap_tokens(r["turns"][i]) for r in runs] for i in range(n_turns)]
        avg = [statistics.mean(t) for t in per_turn]
        std = [statistics.stdev(t) if len(t) > 1 else 0 for t in per_turn]
        x = range(1, n_turns+1)
        ax.plot(x, avg, color=COLORS[name], linewidth=2, label=LABELS[name], marker="o", markersize=4)
        ax.fill_between(x, [a-s for a,s in zip(avg,std)], [a+s for a,s in zip(avg,std)], color=COLORS[name], alpha=0.1)

    ax.axhline(y=token_window, color="black", linestyle="--", alpha=0.5, label=f"TOKEN_WINDOW ({token_window})")
    ax.axhline(y=token_window * trigger_threshold, color="gray", linestyle=":", alpha=0.5, label=f"Trigger ({trigger_threshold:.0%})")
    ax.set(xlabel="Turn", ylabel="Context Tokens (approx)", title="Context Window Size by Strategy")
    ax.legend(loc="upper left"); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_phase_accuracy(strategies, save_path="results/phase_accuracy.png"):
    phase_names = ["Phase 1\n(Fact Inject)", "Phase 2\n(Single Dep)", "Phase 3\n(Multi Dep)", "Phase 4\n(Recall)"]
    phase_keys = ["fact_injection", "single_dep", "multi_dep", "recall"]
    x = np.arange(len(phase_names))
    width = 0.18

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, (name, runs) in enumerate(strategies.items()):
        vals = [statistics.mean([r["phase_scores"][k] for r in runs]) for k in phase_keys]
        ax.bar(x + i * width, vals, width, label=LABELS[name], color=COLORS[name], alpha=0.8)

    ax.set(xlabel="Phase", ylabel="Accuracy (avg)", title="Accuracy by Difficulty Phase", ylim=(0, 1.1))
    ax.set_xticks(x + width * 1.5); ax.set_xticklabels(phase_names)
    ax.legend(); ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_accuracy_heatmap(strategies, n_turns, save_path="results/accuracy_heatmap.png"):
    strategy_names = list(strategies.keys())
    fig, ax = plt.subplots(figsize=(14, 4))
    heatmap_data = np.array([
        [statistics.mean([r["turns"][i]["score"] for r in strategies[name]]) for i in range(n_turns)]
        for name in strategy_names
    ])
    im = ax.imshow(heatmap_data, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(n_turns)); ax.set_xticklabels(range(1, n_turns+1))
    ax.set_yticks(range(len(strategy_names))); ax.set_yticklabels([LABELS[s] for s in strategy_names])
    ax.set(xlabel="Turn", title="Accuracy Heatmap")

    for i in range(len(strategy_names)):
        for j in range(n_turns):
            v = heatmap_data[i, j]
            ax.text(j, i, f"{v:.1f}", ha="center", va="center", color="white" if v < 0.4 or v > 0.8 else "black", fontsize=7)

    plt.colorbar(im, ax=ax, label="Accuracy")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_snapshot_inspector(strategies, turns, inspect_turns):
    strategy_names = list(strategies.keys())
    for turn_num in inspect_turns:
        t = turns[turn_num - 1]
        print(f"\n{'='*70}")
        print(f"T{turn_num}: {t['question'][:80]}...")
        print(f"depends on: {t['depends_on']}")
        for name in strategy_names:
            td = strategies[name][0]["turns"][turn_num - 1]
            snap = td.get("snapshot")
            print(f"\n  {LABELS[name]}: score={td['score']:.1f}  answer={str(td['answer'] or '')[:80]}")
            if snap:
                print(f"    msgs={snap['message_count']}  tokens={snap['total_tokens_in_context']:,}  trimmed={snap['trimmed_count']}")
                for field in ["summary", "scratchpad"]:
                    if snap.get(field):
                        print(f"    {field}: {str(snap[field])[:120]}")


def plot_tradeoff_scatter(strategies, save_path="results/tradeoff_scatter.png"):
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, runs in strategies.items():
        avg_tokens = statistics.mean([
            statistics.mean([_snap_tokens(t) for t in r["turns"]]) for r in runs])
        avg_acc = statistics.mean([r['mean_score'] for r in runs])
        ax.scatter(avg_tokens, avg_acc, color=COLORS[name], s=200, zorder=5)
        ax.annotate(LABELS[name], (avg_tokens, avg_acc), textcoords='offset points',
                    xytext=(10, 5), fontsize=10, color=COLORS[name])
    ax.set(xlabel='Avg Context Tokens per Turn', ylabel='Mean Accuracy',
           title='Tradeoff: Context Size vs Accuracy', ylim=(0, 1.05))
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_action_timeline(strategies, n_turns, save_path="results/action_timeline.png"):
    strategy_names = list(strategies.keys())
    action_colors = {'pass': '#cccccc', 'trimmed': '#e67e22', 'summarized': '#27ae60', 'scratchpad': '#2980b9'}
    fig, axes = plt.subplots(len(strategy_names), 1, figsize=(14, 2*len(strategy_names)), sharex=True)
    for ax, name in zip(axes, strategy_names):
        run = strategies[name][0]
        for t in run['turns']:
            snap = t.get('snapshot') or {}
            trimmed = snap.get('trimmed_count', 0)
            has_summary = bool(snap.get('summary'))
            has_scratchpad = bool(snap.get('scratchpad'))
            if trimmed and has_summary: c = action_colors['summarized']
            elif trimmed: c = action_colors['trimmed']
            elif has_scratchpad: c = action_colors['scratchpad']
            else: c = action_colors['pass']
            ax.barh(0, 1, left=t['turn']-1, color=c, edgecolor='white', linewidth=0.5)
            ax.text(t['turn']-0.5, 0, f"{t['score']:.0f}", ha='center', va='center', fontsize=7,
                    color='white' if c != '#cccccc' else 'black')
        ax.set_yticks([]); ax.set_ylabel(LABELS[name], rotation=0, ha='right', fontsize=9)
        ax.set_xlim(0, n_turns)
    axes[-1].set_xticks(range(n_turns)); axes[-1].set_xticklabels(range(1, n_turns+1))
    axes[-1].set_xlabel('Turn')
    axes[0].set_title('Strategy Actions (gray=pass, orange=trim, green=summarize, blue=scratchpad)')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_cumulative_cost(strategies, n_turns, save_path="results/cumulative_cost.png"):
    fig, ax = plt.subplots(figsize=(12, 6))
    for name, runs in strategies.items():
        avg_cumcost = []
        for i in range(n_turns):
            cumcosts = [sum(r['turns'][j]['input_tokens'] * INPUT_COST + r['turns'][j]['output_tokens'] * OUTPUT_COST
                           for j in range(i+1)) for r in runs]
            avg_cumcost.append(statistics.mean(cumcosts))
        ax.plot(range(1, n_turns+1), [c * 1000 for c in avg_cumcost],
                color=COLORS[name], linewidth=2, label=LABELS[name], marker='o', markersize=4)
    ax.set(xlabel='Turn', ylabel='Cumulative Cost (x $0.001)', title='Cumulative Cost by Strategy')
    ax.legend(loc='upper left'); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_radar(strategies, save_path="results/radar.png"):
    categories = ['Accuracy', 'Retention', 'Token\nEfficiency', 'Cost\nEfficiency', 'P4\n(Recall)']
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    max_tokens = max(statistics.mean([r['total_input_tokens'] for r in runs]) for runs in strategies.values())
    max_cost = max(statistics.mean([r['total_cost'] for r in runs]) for runs in strategies.values())

    for name, runs in strategies.items():
        acc = statistics.mean([r['mean_score'] for r in runs])
        ret = statistics.mean([r['retention_score'] for r in runs])
        tok_eff = 1 - (statistics.mean([r['total_input_tokens'] for r in runs]) / max_tokens) if max_tokens > 0 else 0
        cost_eff = 1 - (statistics.mean([r['total_cost'] for r in runs]) / max_cost) if max_cost > 0 else 0
        p3 = statistics.mean([r['phase_scores']['recall'] for r in runs])
        values = [acc, ret, tok_eff, cost_eff, p3] + [acc]
        ax.plot(angles, values, color=COLORS[name], linewidth=2, label=LABELS[name])
        ax.fill(angles, values, color=COLORS[name], alpha=0.1)

    ax.set_xticks(angles[:-1]); ax.set_xticklabels(categories, fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_title('Strategy Comparison Radar', y=1.08)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
