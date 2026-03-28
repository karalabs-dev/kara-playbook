import os, getpass


def set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f'{var}: ')


def print_stats(stats, answer=None):
    header = f"{'Loop':<6} {'In':<8} {'Out':<8} {'Tool?':<7} {'Details'}"
    print(header)
    print('-' * len(header))
    for s in stats:
        calls = s.get('calls')
        query = s.get('query', '')
        if calls:
            detail = ', '.join(f"{c['name']}({c['args']})" for c in calls)
        elif query:
            detail = f'query: {query}'
        else:
            detail = '—'
        tc = 'Y' if s['tool_call'] else '—'
        print(f"{s['loop']:<6} {s['input_tokens']:<8} {s['output_tokens']:<8} {tc:<7} {detail}")

    total = sum(s['input_tokens'] + s['output_tokens'] for s in stats)
    print(f"\nTotal tokens: {total}")
    if answer:
        print(f"Answer: {answer[:300]}")


def total_tokens(stats):
    return sum(s['input_tokens'] + s['output_tokens'] for s in stats)


# gemini-3.1-flash-lite-preview pricing
INPUT_COST = 0.075 / 1_000_000
OUTPUT_COST = 0.30 / 1_000_000


def cost_usd(stats):
    inp = sum(s['input_tokens'] for s in stats)
    out = sum(s['output_tokens'] for s in stats)
    return inp * INPUT_COST + out * OUTPUT_COST


def print_summary(task_stats):
    rows = [
        ('Sequential math',              task_stats[1], 'Deterministic'),
        ('Constraint sat. (solution)',    task_stats[2], 'Model finds answer'),
        ('Constraint sat. (no solution)', task_stats[3], 'Never converges'),
        ('Search (ambiguous)',            task_stats[4], 'Model decides'),
    ]
    header = f"{'Task':<34} {'Loops':<7} {'Tokens':<8} {'Cost':<12} {'Exit'}"
    print(header)
    print('-' * len(header))
    for name, stats, exit_cond in rows:
        print(f"{name:<34} {len(stats):<7} {total_tokens(stats):<8} ${cost_usd(stats):.6f}   {exit_cond}")

    den = total_tokens(task_stats[2])
    if den > 0:
        ratio = total_tokens(task_stats[3]) / den
        print(f'\nNo-solution task used {ratio:.1f}x the tokens of the solution task.')
