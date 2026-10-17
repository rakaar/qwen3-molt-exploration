"""Analyze saved teacher-forced losses and plot train/validation curves locally."""
import argparse
import csv
import json
from pathlib import Path


def paired_bootstrap(rows_by_step, draws=2000):
    """Use the same resampled example indices for every checkpoint comparison."""
    import numpy as np

    first = next(iter(rows_by_step.values()))
    identities = [r['dataset_index'] for r in first]
    counts = np.array([r['response_tokens'] for r in first], dtype=float)
    if len(identities) < 2 or len(set(identities)) != len(identities) or (counts <= 0).any():
        raise ValueError('Distinct examples with positive token counts required')
    sample = np.random.default_rng(20261017).integers(0, len(first), size=(draws, len(first)))
    denominator = counts[sample].sum(axis=1)
    result = {}
    for step, rows in rows_by_step.items():
        if ([r['dataset_index'] for r in rows] != identities
                or [r['response_tokens'] for r in rows] != counts.tolist()):
            raise ValueError('Paired comparisons require identical examples and response-token masks')
        sums = np.array([r['loss_sum'] for r in rows])
        if not np.isfinite(sums).all():
            raise ValueError('Finite response loss required')
        result[step] = sums[sample].sum(axis=1) / denominator
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    directory = args.directory.resolve()
    if directory.is_relative_to(Path(__file__).resolve().parents[2]):
        raise ValueError('Private per-example measurements must remain outside Git')
    summary = json.loads((directory / 'summary.json').read_text())
    if not summary['complete'] or summary['generation_count'] != 0 or summary['optimizer_updates'] != 0:
        raise ValueError('Completed loss-only validation required')
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    checkpoints = summary['checkpoints']
    steps = [r['step'] for r in checkpoints]
    assert steps == [0, 100, 200, 300, 397]
    bootstrap = {}
    for split in ('validation', 'train_subset'):
        rows = {step: json.loads((directory / f'step-{step}-{split}-examples.json').read_text())
                for step in steps}
        bootstrap[split] = paired_bootstrap(rows)
    intervals = {split: {str(step): np.quantile(values, [.025, .975]).tolist()
                         for step, values in series.items()} for split, series in bootstrap.items()}
    adjacent = []
    for before, after in zip(steps[:-1], steps[1:]):
        series = bootstrap['validation'][after] - bootstrap['validation'][before]
        point = next(x for x in checkpoints if x['step'] == after)['validation']['response_token_loss']
        point -= next(x for x in checkpoints if x['step'] == before)['validation']['response_token_loss']
        adjacent.append({'before_step': before, 'after_step': after,
                         'validation_loss_change': point,
                         'exploratory_95_percent_paired_row_bootstrap': np.quantile(series, [.025, .975]).tolist()})
    report = {'best_measured_validation_step': summary['best_measured_validation_step'],
              'checkpoints': checkpoints, 'intervals': intervals, 'adjacent_validation_changes': adjacent,
              'uncertainty_scope': 'Exploratory row bootstrap on this split, conditional on one training seed. Does not measure EM or establish the best number of epochs.',
              'source_summary': str(directory / 'summary.json')}
    (directory / 'analysis.json').write_text(json.dumps(report, indent=2))
    with (directory / 'loss-curve.csv').open('w') as stream:
        writer = csv.writer(stream)
        writer.writerow(['update', 'train_subset_response_loss', 'validation_response_loss', 'validation_minus_train'])
        for row in checkpoints:
            train = row['train_subset']['response_token_loss']
            val = row['validation']['response_token_loss']
            writer.writerow([row['step'], train, val, val - train])
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10})
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.6), layout='constrained')
    fig.suptitle('Qwen3-4B: medical training and validation loss', fontsize=14, fontweight='bold')
    for ax, selected, title in zip(axes, (checkpoints, checkpoints[1:]),
                                   ('Whole epoch, including unchanged base', 'Saved checkpoints: closer view')):
        for split, label, color in [('train_subset', 'Fixed training subset', '#2469a0'),
                                     ('validation', 'Held-out validation', '#c24c35')]:
            x = [r['step'] for r in selected]
            y = [r[split]['response_token_loss'] for r in selected]
            low = [intervals[split][str(r['step'])][0] for r in selected]
            high = [intervals[split][str(r['step'])][1] for r in selected]
            ax.plot(x, y, marker='o', linewidth=2, markersize=5, label=label, color=color)
            ax.fill_between(x, low, high, color=color, alpha=.12, linewidth=0)
        ax.set(title=title, xlabel='Training updates', ylabel='Response-token cross-entropy loss')
        ax.grid(alpha=.18)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].legend(frameon=False)
    fig.text(.5, -.035,
             '705 held-out examples; fixed 705 training examples. Shading: exploratory 95% row-bootstrap intervals.\n'
             'Same Qwen3-4B weights and original split. No new generated answers or training updates.',
             ha='center', fontsize=9, color='#444444')
    fig.savefig(directory / 'loss-curve.png', dpi=180, bbox_inches='tight')
    fig.savefig(directory / 'loss-curve.svg', bbox_inches='tight')
    plt.close(fig)
    print(json.dumps({'best_measured_validation_step': report['best_measured_validation_step'],
                      'adjacent_validation_changes': adjacent,
                      'figure': str(directory / 'loss-curve.png')}, indent=2))


if __name__ == '__main__':
    main()
