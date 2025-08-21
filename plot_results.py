"""
Script for plotting and analyzing experiment results from AMAL vs MAPPO runs.
"""

import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Настраиваем стиль графиков для красивого вида
sns.set_theme(style="whitegrid")

def load_results(exp_dir: Path) -> pd.DataFrame:
    """Загружает все .csv логи из директории эксперимента в один DataFrame."""
    all_files = exp_dir.glob("**/*.csv")
    df_list = []
    for f in all_files:
        try:
            df = pd.read_csv(f)
            # Извлекаем параметры из имени файла/пути
            parts = f.stem.split('_')
            df['algorithm'] = parts[0]
            df['scenario'] = parts[1]
            df['seed'] = int(parts[2].replace('seed', ''))
            df_list.append(df)
        except Exception as e:
            print(f"Could not read file {f}: {e}")
    if not df_list:
        return pd.DataFrame()
    return pd.concat(df_list, ignore_index=True)

def plot_learning_curves(df: pd.DataFrame, output_dir: Path):
    """Строит кривые обучения (win rate) для каждого сценария."""
    for scenario, data in df.groupby('scenario'):
        plt.figure(figsize=(10, 6))
        
        sns.lineplot(data=data, x='episode', y='battle_won', hue='algorithm', errorbar='sd')
        
        plt.title(f'Win Rate on {scenario}')
        plt.xlabel('Episode')
        plt.ylabel('Win Rate')
        plt.legend(title='Algorithm')
        plt.grid(True)
        
        output_path = output_dir / f'learning_curve_{scenario}.png'
        plt.savefig(output_path)
        plt.close()
        print(f"Saved learning curve for {scenario} to {output_path}")

def plot_sample_efficiency(df: pd.DataFrame, output_dir: Path, target_win_rate=0.8):
    """Сравнивает sample efficiency: количество шагов до достижения целевого win rate."""
    efficiency_data = []
    for (algo, scenario), data in df.groupby(['algorithm', 'scenario']):
        # Сглаживаем кривую для стабильности
        smooth_win_rate = data.groupby('episode')['battle_won'].mean().rolling(window=10, min_periods=1).mean()
        
        # Находим первый эпизод, где достигнут целевой win rate
        achieved_episodes = smooth_win_rate[smooth_win_rate >= target_win_rate]
        if not achieved_episodes.empty:
            steps_to_target = achieved_episodes.index[0]
            efficiency_data.append({'algorithm': algo, 'scenario': scenario, 'episodes_to_target': steps_to_target})

    if not efficiency_data:
        print("Could not compute sample efficiency (target win rate not reached).")
        return

    efficiency_df = pd.DataFrame(efficiency_data)
    
    plt.figure(figsize=(12, 7))
    sns.barplot(data=efficiency_df, x='scenario', y='episodes_to_target', hue='algorithm')
    
    plt.title(f'Sample Efficiency (Episodes to Reach {target_win_rate*100}% Win Rate)')
    plt.xlabel('Scenario')
    plt.ylabel('Episodes')
    
    output_path = output_dir / 'sample_efficiency.png'
    plt.savefig(output_path)
    plt.close()
    print(f"Saved sample efficiency plot to {output_path}")


def create_summary_table(df: pd.DataFrame, output_dir: Path):
    """Создает и сохраняет сводную таблицу с финальными результатами."""
    # Берем средний win rate за последние 10% эпизодов
    last_10_percent_episode = df['episode'].max() * 0.9
    final_performance = df[df['episode'] >= last_10_percent_episode]
    
    summary = final_performance.groupby(['algorithm', 'scenario'])['battle_won'].agg(['mean', 'std']).unstack()
    
    # Форматируем для красивого вывода
    summary_styled = summary.style.format("{:.2f} ± {:.2f}", na_rep="-", subset=pd.IndexSlice[:, pd.IndexSlice[:, 'mean':'std']])
    
    print("\n--- Final Performance Summary ---")
    print(summary)
    
    # Сохраняем в LaTeX и CSV
    output_tex = output_dir / 'summary_table.tex'
    output_csv = output_dir / 'summary_table.csv'
    
    summary.to_csv(output_csv)
    with open(output_tex, 'w') as f:
        f.write(summary_styled.to_latex())

    print(f"Saved summary table to {output_csv} and {output_tex}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_dir', type=str, required=True, help="Directory containing experiment log files (.csv).")
    args = parser.parse_args()

    exp_path = Path(args.exp_dir)
    if not exp_path.exists():
        print(f"Error: Directory not found at {exp_path}")
        return

    output_path = exp_path / "plots"
    output_path.mkdir(exist_ok=True)

    df = load_results(exp_path)
    if df.empty:
        print("No valid result files found. Exiting.")
        return

    plot_learning_curves(df, output_path)
    plot_sample_efficiency(df, output_path)
    create_summary_table(df, output_path)

if __name__ == '__main__':
    main()
