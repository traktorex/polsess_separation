#!/usr/bin/env python3
"""
Comprehensive analysis of DPRNN Stage 3 sweep results.
Selects top 5 configurations for final validation based on multiple criteria.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# File paths
HYPERBAND_CSV = "results/wandb_export_2026-02-01T17_38_30.533+01_00_8k.csv"
CONSERVATIVE_CSV = "results/wandb_export_2026-02-01T17_47_16.789+01_00_8kcon.csv"

def load_and_clean_data(hyperband_path, conservative_path):
    """Load both CSV files and combine them."""
    # Load data
    df_hyperband = pd.read_csv(hyperband_path)
    df_conservative = pd.read_csv(conservative_path)
    
    # Add sweep type column
    df_hyperband['sweep_type'] = 'hyperband'
    df_conservative['sweep_type'] = 'conservative'
    
    # Combine
    df = pd.concat([df_hyperband, df_conservative], ignore_index=True)
    
    # Filter for finished runs only
    df_finished = df[df['State'] == 'finished'].copy()
    
    print(f"📊 Total runs: {len(df)}")
    print(f"   ✅ Finished: {len(df_finished)}")
    print(f"   ❌ Killed/Failed: {len(df) - len(df_finished)}")
    print(f"\nSweep breakdown:")
    print(f"   Hyperband: {len(df_hyperband)} ({(df_hyperband['State']=='finished').sum()} finished)")
    print(f"   Conservative: {len(df_conservative)} ({(df_conservative['State']=='finished').sum()} finished)")
    
    return df_finished

def analyze_performance(df):
    """Analyze performance metrics."""
    print("\n" + "="*60)
    print("📈 PERFORMANCE ANALYSIS")
    print("="*60)
    
    # Overall statistics
    print(f"\nValidation SI-SDR Statistics:")
    print(f"   Best:   {df['best_val_sisdr'].max():.4f} dB")
    print(f"   Mean:   {df['best_val_sisdr'].mean():.4f} dB")
    print(f"   Median: {df['best_val_sisdr'].median():.4f} dB")
    print(f"   Std:    {df['best_val_sisdr'].std():.4f} dB")
    
    # Compare sweeps
    print(f"\nBy Sweep Type:")
    for sweep_type in ['hyperband', 'conservative']:
        subset = df[df['sweep_type'] == sweep_type]
        print(f"   {sweep_type.capitalize()}:")
        print(f"      Best:  {subset['best_val_sisdr'].max():.4f} dB")
        print(f"      Mean:  {subset['best_val_sisdr'].mean():.4f} dB")
        print(f"      Count: {len(subset)}")
    
    return df

def check_hyperparameter_diversity(df, top_n=10):
    """Check if top configurations are diverse in hyperparameter space."""
    print("\n" + "="*60)
    print(f"🔍 HYPERPARAMETER DIVERSITY (Top {top_n})")
    print("="*60)
    
    top = df.nlargest(top_n, 'best_val_sisdr')
    
    params = ['lr', 'weight_decay', 'grad_clip_norm', 'lr_factor', 'lr_patience']
    
    for param in params:
        if param in top.columns:
            print(f"\n{param}:")
            print(f"   Range: {top[param].min():.2e} to {top[param].max():.2e}")
            print(f"   Mean:  {top[param].mean():.2e}")
            print(f"   Std:   {top[param].std():.2e}")
            print(f"   Unique values: {top[param].nunique()}")

def select_top_configs(df, n=5, diversity_weight=0.2):
    """
    Select top N configurations balancing performance and diversity.
    
    Strategy:
    1. Rank by best_val_sisdr (primary metric)
    2. Consider hyperparameter diversity to avoid redundancy
    3. Prefer configs that finished training (completed more epochs)
    """
    print("\n" + "="*60)
    print(f"🎯 SELECTING TOP {n} CONFIGURATIONS")
    print("="*60)
    
    # Sort by performance
    df_sorted = df.sort_values('best_val_sisdr', ascending=False).reset_index(drop=True)
    
    # Strategy 1: Pure performance (baseline)
    top_performance = df_sorted.head(10)
    
    print(f"\nTop 10 by Performance:")
    for idx, row in top_performance.iterrows():
        early_stop = "⚠️ EARLY" if row.get('early_stopped', 0) == 1 else "✓"
        print(f"   {idx+1}. {row['Name']:20s} | SI-SDR: {row['best_val_sisdr']:.4f} dB | "
              f"Epoch: {row.get('epoch', 'N/A'):>2} | {early_stop} | {row['sweep_type']}")
    
    # Strategy 2: Performance + diversity
    selected = []
    selected_params = []
    
    param_cols = ['lr', 'weight_decay', 'grad_clip_norm', 'lr_factor', 'lr_patience']
    
    for idx, row in df_sorted.iterrows():
        if len(selected) >= n:
            break
        
        if len(selected) == 0:
            # Always take the best
            selected.append(row)
            selected_params.append(row[param_cols].values)
            continue
        
        # Calculate diversity score (Euclidean distance to selected configs)
        current_params = row[param_cols].values
        
        # Normalize parameters for distance calculation
        normalized_current = (current_params - df[param_cols].mean()) / df[param_cols].std()
        
        min_distance = float('inf')
        for prev_params in selected_params:
            normalized_prev = (prev_params - df[param_cols].mean()) / df[param_cols].std()
            distance = np.linalg.norm(normalized_current - normalized_prev)
            min_distance = min(min_distance, distance)
        
        # Diversity bonus: prefer configs that are farther from already selected
        # But still heavily weight performance
        performance_rank = idx + 1
        diversity_bonus = min_distance * diversity_weight * 10
        
        # If performance is very close to top and diverse, include it
        if performance_rank <= n * 3:  # Consider top 15 for 5 selections
            selected.append(row)
            selected_params.append(current_params)
            print(f"\n   Selected #{len(selected)}: {row['Name']}")
            print(f"      SI-SDR: {row['best_val_sisdr']:.4f} dB (rank: {performance_rank})")
            print(f"      Diversity score: {min_distance:.2f}")
    
    return pd.DataFrame(selected)

def create_summary_table(selected_df):
    """Create a detailed summary table of selected configurations."""
    print("\n" + "="*60)
    print("📋 SELECTED CONFIGURATIONS FOR FINAL VALIDATION")
    print("="*60)
    
    cols_to_show = ['Name', 'best_val_sisdr', 'lr', 'weight_decay', 'grad_clip_norm', 
                    'lr_factor', 'lr_patience', 'epoch', 'early_stopped', 'sweep_type']
    
    print("\n")
    for idx, row in selected_df.iterrows():
        print(f"\n{'='*60}")
        print(f"CONFIG #{idx+1}: {row['Name']}")
        print(f"{'='*60}")
        print(f"Performance:")
        print(f"   Validation SI-SDR:     {row['best_val_sisdr']:.4f} dB")
        print(f"   Training SI-SDR:       {row.get('train_si_sdr', 'N/A'):.4f} dB")
        print(f"   Final Epoch:           {row.get('epoch', 'N/A')}")
        print(f"   Early Stopped:         {'Yes' if row.get('early_stopped', 0) == 1 else 'No'}")
        print(f"   Sweep Type:            {row['sweep_type']}")
        
        print(f"\nHyperparameters:")
        print(f"   Learning Rate:         {row['lr']:.6f}")
        print(f"   Weight Decay:          {row['weight_decay']:.2e}")
        print(f"   Grad Clip Norm:        {row['grad_clip_norm']:.2f}")
        print(f"   LR Factor:             {row['lr_factor']:.4f}")
        print(f"   LR Patience:           {int(row['lr_patience'])}")
        
        print(f"\nFinal Learning Rate:     {row.get('train_lr', 'N/A'):.2e}")
    
    # Export to CSV
    output_file = "results/top5_configs_for_validation.csv"
    selected_df[cols_to_show].to_csv(output_file, index=False)
    print(f"\n\n💾 Selected configurations saved to: {output_file}")
    
    return selected_df

def create_config_yaml_snippets(selected_df):
    """Generate YAML config snippets for easy validation runs."""
    print("\n" + "="*60)
    print("⚙️  CONFIGURATION SNIPPETS FOR VALIDATION")
    print("="*60)
    
    output_file = "results/validation_configs.txt"
    
    with open(output_file, 'w') as f:
        f.write("# Top 5 DPRNN Configurations for Final Validation (16K samples, 3 seeds)\n")
        f.write("# Run each config with seeds: 42, 123, 456\n\n")
        
        for idx, row in selected_df.iterrows():
            config_name = f"config{idx+1}_{row['Name']}"
            
            snippet = f"""
{'='*70}
# Configuration {idx+1}: {row['Name']}
# Expected Performance: {row['best_val_sisdr']:.4f} dB (on 8K samples)
{'='*70}

training:
  lr: {row['lr']:.8f}
  weight_decay: {row['weight_decay']:.2e}
  grad_clip_norm: {row['grad_clip_norm']:.2f}
  lr_factor: {row['lr_factor']:.4f}
  lr_patience: {int(row['lr_patience'])}
  num_epochs: 80
  early_stopping_patience: 15

# Run command:
# python train.py --config experiments/dprnn/dprnn_16000.yaml \\
#   --lr {row['lr']:.8f} \\
#   --weight-decay {row['weight_decay']:.2e} \\
#   --grad-clip-norm {row['grad_clip_norm']:.2f} \\
#   --lr-factor {row['lr_factor']:.4f} \\
#   --lr-patience {int(row['lr_patience'])} \\
#   --seed {{SEED}}

"""
            f.write(snippet)
            print(snippet)
    
    print(f"\n💾 Config snippets saved to: {output_file}")

def plot_analysis(df, selected_df):
    """Create visualization plots."""
    print("\n" + "="*60)
    print("📊 CREATING VISUALIZATIONS")
    print("="*60)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. SI-SDR distribution by sweep type
    ax = axes[0, 0]
    df.boxplot(column='best_val_sisdr', by='sweep_type', ax=ax)
    ax.set_title('SI-SDR Distribution by Sweep Type')
    ax.set_xlabel('Sweep Type')
    ax.set_ylabel('Best Validation SI-SDR (dB)')
    plt.sca(ax)
    plt.xticks(rotation=0)
    
    # 2. Top configs comparison
    ax = axes[0, 1]
    top20 = df.nlargest(20, 'best_val_sisdr')
    colors = ['green' if name in selected_df['Name'].values else 'gray' 
              for name in top20['Name']]
    ax.barh(range(len(top20)), top20['best_val_sisdr'], color=colors)
    ax.set_yticks(range(len(top20)))
    ax.set_yticklabels(top20['Name'], fontsize=8)
    ax.set_xlabel('Best Validation SI-SDR (dB)')
    ax.set_title('Top 20 Configs (Green = Selected for Validation)')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)
    
    # 3. Learning rate vs performance
    ax = axes[1, 0]
    scatter = ax.scatter(df['lr'], df['best_val_sisdr'], 
                         c=df['weight_decay'], cmap='viridis', 
                         alpha=0.6, s=50)
    ax.scatter(selected_df['lr'], selected_df['best_val_sisdr'], 
               color='red', s=200, marker='*', edgecolors='black', 
               linewidths=1.5, label='Selected', zorder=5)
    ax.set_xscale('log')
    ax.set_xlabel('Learning Rate')
    ax.set_ylabel('Best Validation SI-SDR (dB)')
    ax.set_title('LR vs Performance (color = weight decay)')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.colorbar(scatter, ax=ax, label='Weight Decay')
    
    # 4. Hyperparameter correlations
    ax = axes[1, 1]
    param_cols = ['lr', 'weight_decay', 'grad_clip_norm', 'lr_factor']
    correlation = df[param_cols + ['best_val_sisdr']].corr()['best_val_sisdr'][:-1]
    correlation.plot(kind='bar', ax=ax, color='skyblue', edgecolor='black')
    ax.set_title('Hyperparameter Correlation with SI-SDR')
    ax.set_ylabel('Correlation Coefficient')
    ax.set_xlabel('Hyperparameter')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    output_file = 'results/analysis_plots.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"   Saved plots to: {output_file}")
    plt.close()

def main():
    """Main analysis pipeline."""
    print("\n" + "🧠 DPRNN STAGE 3 SWEEP ANALYSIS 🧠".center(60, "="))
    print()
    
    # Load data
    df = load_and_clean_data(HYPERBAND_CSV, CONSERVATIVE_CSV)
    
    # Analyze performance
    df = analyze_performance(df)
    
    # Check diversity in top configs
    check_hyperparameter_diversity(df, top_n=10)
    
    # Select top 5 configurations
    selected_df = select_top_configs(df, n=5, diversity_weight=0.2)
    
    # Create summary table
    create_summary_table(selected_df)
    
    # Generate config snippets
    create_config_yaml_snippets(selected_df)
    
    # Create plots
    plot_analysis(df, selected_df)
    
    print("\n" + "="*60)
    print("✅ ANALYSIS COMPLETE")
    print("="*60)
    print("\nNext Steps:")
    print("1. Review selected configurations above")
    print("2. Create dprnn_16000.yaml config file")
    print("3. Run each of the 5 configs with 3 seeds (42, 123, 456)")
    print("4. Total validation runs: 15 (5 configs × 3 seeds)")
    print("5. Select final config based on mean performance across seeds")
    print("\n")

if __name__ == "__main__":
    main()
