import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

def load_all_results(results_dir='results'):
    """Load all metrics_summary.csv files from results directory"""
    results = []

    for model_dir in Path(results_dir).iterdir():
        if model_dir.is_dir():
            metrics_file = model_dir / 'metrics_summary.csv'
            if metrics_file.exists():
                df = pd.read_csv(metrics_file)
                model_name = model_dir.name
                df['model_task'] = model_name

                # Parse model and task
                parts = model_name.rsplit('_', 1)
                if len(parts) == 2:
                    df['model'] = parts[0]
                    df['task'] = parts[1]
                else:
                    df['model'] = model_name
                    df['task'] = 'unknown'

                results.append(df)

    if not results:
        print("No results found!")
        return None

    return pd.concat(results, ignore_index=True)

def plot_model_comparison(df, metric='accuracy', save_dir='results/analysis'):
    """Plot comparison of models for a specific metric"""
    os.makedirs(save_dir, exist_ok=True)

    # Separate by task
    for task in df['task'].unique():
        task_df = df[df['task'] == task]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Train vs Test comparison
        test_df = task_df[task_df['split'] == 'test'].sort_values(metric, ascending=False)

        models = test_df['model'].values
        test_scores = test_df[metric].values

        # Get corresponding train scores
        train_scores = []
        for model in models:
            train_score = task_df[(task_df['model'] == model) & (task_df['split'] == 'train')][metric].values
            train_scores.append(train_score[0] if len(train_score) > 0 else 0)

        x = np.arange(len(models))
        width = 0.35

        ax1.bar(x - width/2, train_scores, width, label='Train', alpha=0.8)
        ax1.bar(x + width/2, test_scores, width, label='Test', alpha=0.8)
        ax1.set_xlabel('Model')
        ax1.set_ylabel(metric.capitalize())
        ax1.set_title(f'{metric.capitalize()} Comparison - {task.capitalize()} Task')
        ax1.set_xticks(x)
        ax1.set_xticklabels(models, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)

        # Overfitting analysis
        overfit = np.array(train_scores) - np.array(test_scores)
        colors = ['red' if x > 0.1 else 'green' for x in overfit]

        ax2.bar(x, overfit, color=colors, alpha=0.6)
        ax2.axhline(y=0, color='black', linestyle='--', linewidth=0.8)
        ax2.axhline(y=0.1, color='red', linestyle='--', linewidth=0.8, alpha=0.5)
        ax2.set_xlabel('Model')
        ax2.set_ylabel('Train - Test Gap')
        ax2.set_title(f'Overfitting Analysis - {task.capitalize()} Task')
        ax2.set_xticks(x)
        ax2.set_xticklabels(models, rotation=45, ha='right')
        ax2.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{save_dir}/{metric}_comparison_{task}.png', dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Saved: {metric}_comparison_{task}.png")

def plot_all_metrics_heatmap(df, save_dir='results/analysis'):
    """Create heatmap of all metrics for all models"""
    os.makedirs(save_dir, exist_ok=True)

    metrics = ['accuracy', 'precision', 'recall', 'f1']

    for task in df['task'].unique():
        # Get test set metrics
        task_df = df[(df['task'] == task) & (df['split'] == 'test')]

        # Pivot to get models x metrics
        pivot_df = task_df.pivot_table(
            values=metrics,
            index='model',
            aggfunc='first'
        )

        plt.figure(figsize=(10, 6))
        sns.heatmap(pivot_df, annot=True, fmt='.4f', cmap='YlOrRd',
                   cbar_kws={'label': 'Score'})
        plt.title(f'All Metrics Heatmap - {task.capitalize()} Task (Test Set)')
        plt.xlabel('Metric')
        plt.ylabel('Model')
        plt.tight_layout()
        plt.savefig(f'{save_dir}/metrics_heatmap_{task}.png', dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Saved: metrics_heatmap_{task}.png")

def generate_summary_table(df, save_dir='results/analysis'):
    """Generate summary table with best models for each metric"""
    os.makedirs(save_dir, exist_ok=True)

    metrics = ['accuracy', 'precision', 'recall', 'f1']

    for task in df['task'].unique():
        task_df = df[(df['task'] == task) & (df['split'] == 'test')]

        print(f"\n{'='*60}")
        print(f"SUMMARY - {task.upper()} TASK")
        print(f"{'='*60}\n")

        summary_data = []

        for metric in metrics:
            best_row = task_df.loc[task_df[metric].idxmax()]
            worst_row = task_df.loc[task_df[metric].idxmin()]

            summary_data.append({
                'Metric': metric.capitalize(),
                'Best Model': best_row['model'],
                'Best Score': f"{best_row[metric]:.4f}",
                'Worst Model': worst_row['model'],
                'Worst Score': f"{worst_row[metric]:.4f}",
                'Range': f"{best_row[metric] - worst_row[metric]:.4f}"
            })

            print(f"{metric.capitalize():12} - Best: {best_row['model']:20} ({best_row[metric]:.4f})")

        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(f'{save_dir}/summary_{task}.csv', index=False)
        print(f"\nSummary saved to: summary_{task}.csv")

        # Full leaderboard
        print(f"\n{'='*60}")
        print(f"LEADERBOARD - {task.upper()} TASK")
        print(f"{'='*60}\n")
        leaderboard = task_df.sort_values('f1', ascending=False)[
            ['model', 'accuracy', 'precision', 'recall', 'f1']
        ]
        print(leaderboard.to_string(index=False))
        leaderboard.to_csv(f'{save_dir}/leaderboard_{task}.csv', index=False)

def main():
    print("Review Classifier - Results Analysis")
    print("=" * 60)

    # Load all results
    df = load_all_results()

    if df is None:
        return

    print(f"\nLoaded results from {len(df['model_task'].unique())} experiments")
    print(f"Tasks: {', '.join(df['task'].unique())}")
    print(f"Models: {', '.join(df['model'].unique())}")

    # Create analysis directory
    os.makedirs('results/analysis', exist_ok=True)

    # Generate all plots
    print("\nGenerating comparison plots...")
    for metric in ['accuracy', 'precision', 'recall', 'f1']:
        plot_model_comparison(df, metric)

    print("\nGenerating heatmaps...")
    plot_all_metrics_heatmap(df)

    print("\nGenerating summary tables...")
    generate_summary_table(df)

    print("\n" + "=" * 60)
    print("Analysis complete! Check results/analysis/ directory")
    print("=" * 60)

if __name__ == '__main__':
    main()
