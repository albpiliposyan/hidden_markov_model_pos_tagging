"""Generate all visualizations for HMM project."""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from hmm import HiddenMarkovModel
from hmm_trigram import TrigramHMM
from data_utils import load_armenian_dataset
from visualization import (
    visualize_model_comparison,
    visualize_suffix_patterns,
    visualize_unknown_word_performance,
    compare_models_on_word_types,
    create_all_visualizations
)


def train_and_evaluate(train_data, test_data, model_name, **kwargs):
    """Train and evaluate a single HMM configuration."""
    print(f"  Training: {model_name}...")
    
    hmm = HiddenMarkovModel(**kwargs)
    hmm.train(train_data)
    
    results = hmm.evaluate(test_data)
    accuracy = results['accuracy'] * 100
    
    print(f"    Accuracy: {accuracy:.2f}%")
    
    return hmm, accuracy


def create_comprehensive_comparison_chart(results, output_dir):
    """Create bar chart comparing all model accuracies."""
    # Set style
    sns.set_style("whitegrid")
    plt.figure(figsize=(16, 8))
    
    # Prepare data
    names = list(results.keys())
    accuracies = list(results.values())
    
    # Define colors for different model types
    colors = []
    for name in names:
        if 'Classical' in name:
            colors.append('#d62728')  # Red for classical
        elif 'Prefix+Suffix' in name:
            colors.append('#2ca02c')  # Green for prefix+suffix
        elif 'Suffix-only' in name:
            colors.append('#1f77b4')  # Blue for suffix-only
        elif 'Prefix-only' in name:
            colors.append('#ff7f0e')  # Orange for prefix-only
        else:
            colors.append('#7f7f7f')  # Gray for others
    
    # Create bar chart
    bars = plt.bar(range(len(names)), accuracies, color=colors, edgecolor='black', linewidth=1.2)
    
    # Add value labels on top of bars
    for i, (bar, acc) in enumerate(zip(bars, accuracies)):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{acc:.2f}%',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Customize plot
    plt.xlabel('Model Configuration', fontsize=12, fontweight='bold')
    plt.ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    plt.title('HMM POS Tagging: Comparison of Different Model Configurations', 
              fontsize=14, fontweight='bold', pad=20)
    plt.xticks(range(len(names)), names, rotation=45, ha='right', fontsize=10)
    plt.ylim([min(accuracies) - 2, max(accuracies) + 2])
    plt.grid(axis='y', alpha=0.3)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#d62728', edgecolor='black', label='Classical HMM'),
        Patch(facecolor='#1f77b4', edgecolor='black', label='Suffix-only'),
        Patch(facecolor='#ff7f0e', edgecolor='black', label='Prefix-only'),
        Patch(facecolor='#2ca02c', edgecolor='black', label='Prefix+Suffix')
    ]
    plt.legend(handles=legend_elements, loc='upper left', fontsize=10)
    
    plt.tight_layout()
    
    # Save figure
    output_file = os.path.join(output_dir, 'model_comparison.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n  Model comparison chart saved to: {output_file}")
    
    plt.close()


def create_word_types_comparison_chart(models_dict, test_data, output_dir):
    """Create grouped bar chart comparing model performance on known vs unknown words."""
    # Evaluate each model on known vs unknown words
    model_results = {}
    
    print("\n  Evaluating models on word types...")
    for model_name, model in models_dict.items():
        print(f"    {model_name}...")
        
        # Get vocabulary
        vocab = set(model.word_to_idx.keys())
        
        # Track predictions for known and unknown words
        known_correct = 0
        known_total = 0
        unknown_correct = 0
        unknown_total = 0
        
        for sentence in test_data:
            words = [word for word, _ in sentence]
            true_tags = [tag for _, tag in sentence]
            predicted_tags = model.predict(words)
            
            for word, true_tag, pred_tag in zip(words, true_tags, predicted_tags):
                # Check if word is known (case-insensitive)
                is_known = (word in vocab or 
                           word.lower() in vocab or 
                           word.capitalize() in vocab)
                
                if is_known:
                    known_total += 1
                    if true_tag == pred_tag:
                        known_correct += 1
                else:
                    unknown_total += 1
                    if true_tag == pred_tag:
                        unknown_correct += 1
        
        known_acc = (known_correct / known_total * 100) if known_total > 0 else 0
        unknown_acc = (unknown_correct / unknown_total * 100) if unknown_total > 0 else 0
        
        model_results[model_name] = {
            'known': known_acc,
            'unknown': unknown_acc
        }
    
    # Create visualization
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(14, 8))
    
    models = list(model_results.keys())
    known_accs = [model_results[m]['known'] for m in models]
    unknown_accs = [model_results[m]['unknown'] for m in models]
    
    x = np.arange(len(models))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, known_accs, width, label='Known Words', 
                   color='#2ecc71', edgecolor='black', linewidth=1.2)
    bars2 = ax.bar(x + width/2, unknown_accs, width, label='Unknown Words',
                   color='#e74c3c', edgecolor='black', linewidth=1.2)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                   f'{height:.1f}%',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_xlabel('Model Configuration', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Model Performance: Known vs Unknown Words', 
                fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right', fontsize=10)
    ax.legend(fontsize=11, loc='lower right')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    output_file = os.path.join(output_dir, 'models_word_types_comparison.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n  Word types comparison saved to: {output_file}")
    
    plt.close()
    
    return model_results


def main():
    """Generate all visualizations."""
    
    print("="*70)
    print("GENERATING ALL HMM VISUALIZATIONS")
    print("="*70)
    
    # Load Data
    print("\n[1] Loading Dataset...")
    train_data, dev_data, test_data = load_armenian_dataset()
    combined_train_data = train_data + dev_data
    
    print(f"  Training: {len(combined_train_data)} sentences")
    print(f"  Test: {len(test_data)} sentences")
    
    # Train all model configurations
    print("\n[2] Training All Model Configurations...")
    
    configurations = [
        {
            'name': 'Classical HMM (no affix)',
            'params': {'use_suffix_model': False, 'use_prefix_model': False}
        },
        {
            'name': 'Suffix-only (n=2)',
            'params': {'use_suffix_model': True, 'suffix_length': 2, 'use_prefix_model': False}
        },
        {
            'name': 'Suffix-only (n=3)',
            'params': {'use_suffix_model': True, 'suffix_length': 3, 'use_prefix_model': False}
        },
        {
            'name': 'Suffix-only (n=4)',
            'params': {'use_suffix_model': True, 'suffix_length': 4, 'use_prefix_model': False}
        },
        {
            'name': 'Prefix-only (n=2)',
            'params': {'use_suffix_model': False, 'use_prefix_model': True, 'prefix_length': 2}
        },
        {
            'name': 'Prefix-only (n=3)',
            'params': {'use_suffix_model': False, 'use_prefix_model': True, 'prefix_length': 3}
        },
        {
            'name': 'Prefix-only (n=4)',
            'params': {'use_suffix_model': False, 'use_prefix_model': True, 'prefix_length': 4}
        },
        {
            'name': 'Prefix+Suffix (pref=2, suff=2)',
            'params': {'use_suffix_model': True, 'suffix_length': 2, 'use_prefix_model': True, 'prefix_length': 2}
        },
        {
            'name': 'Prefix+Suffix (pref=3, suff=3)',
            'params': {'use_suffix_model': True, 'suffix_length': 3, 'use_prefix_model': True, 'prefix_length': 3}
        },
        {
            'name': 'Prefix+Suffix (pref=3, suff=2)',
            'params': {'use_suffix_model': True, 'suffix_length': 2, 'use_prefix_model': True, 'prefix_length': 3}
        },
        {
            'name': 'Prefix+Suffix (pref=2, suff=3)',
            'params': {'use_suffix_model': True, 'suffix_length': 3, 'use_prefix_model': True, 'prefix_length': 2}
        },
        {
            'name': 'Prefix+Suffix (pref=4, suff=4)',
            'params': {'use_suffix_model': True, 'suffix_length': 4, 'use_prefix_model': True, 'prefix_length': 4}
        },
    ]
    
    # Train and evaluate each configuration
    results = {}
    trained_models = {}
    
    for config in configurations:
        model, accuracy = train_and_evaluate(
            combined_train_data, 
            test_data, 
            config['name'], 
            **config['params']
        )
        results[config['name']] = accuracy
        trained_models[config['name']] = model
    
    # Print summary
    print(f"\n{'='*70}")
    print("ACCURACY SUMMARY")
    print(f"{'='*70}")
    for name, accuracy in results.items():
        print(f"{name:<45} {accuracy:>6.2f}%")
    
    # Create output directory
    output_dir = 'visualizations'
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate comprehensive comparison visualization
    print("\n[3] Generating Comprehensive Model Comparison...")
    create_comprehensive_comparison_chart(results, output_dir)
    
    # Generate word types comparison for selected models
    print("\n[4] Generating Word Types Comparison...")
    selected_models = {
        'Classical HMM (no affix)': trained_models['Classical HMM (no affix)'],
        'Suffix-only (n=2)': trained_models['Suffix-only (n=2)'],
        'Suffix-only (n=3)': trained_models['Suffix-only (n=3)'],
        'Suffix-only (n=4)': trained_models['Suffix-only (n=4)'],
        'Prefix-only (n=2)': trained_models['Prefix-only (n=2)'],
        'Prefix+Suffix (pref=2, suff=2)': trained_models['Prefix+Suffix (pref=2, suff=2)'],
        'Prefix+Suffix (pref=3, suff=3)': trained_models['Prefix+Suffix (pref=3, suff=3)'],
        'Prefix+Suffix (pref=2, suff=3)': trained_models['Prefix+Suffix (pref=2, suff=3)'],
    }
    word_type_results = create_word_types_comparison_chart(selected_models, test_data, output_dir)
    
    print("\n  Word Type Accuracy Results:")
    for model_name, accs in word_type_results.items():
        print(f"    {model_name}:")
        print(f"      Known: {accs['known']:.2f}%, Unknown: {accs['unknown']:.2f}%")
    
    # Use the best model for detailed visualizations (Prefix+Suffix pref=2, suff=3)
    best_model = trained_models['Prefix+Suffix (pref=2, suff=3)']
    
    # Generate detailed visualizations for best model
    print("\n[5] Generating Detailed Visualizations (Best Model: Prefix+Suffix pref=2, suff=3)...")
    
    print("  Creating suffix pattern analysis...")
    visualize_suffix_patterns(best_model, top_n=30,
                             save_path=f'{output_dir}/suffix_patterns.png')
    
    print("  Creating unknown word performance analysis...")
    stats = visualize_unknown_word_performance(best_model, test_data,
                                              save_path=f'{output_dir}/unknown_word_performance.png')
    
    print(f"\n  Known words accuracy: {stats['known_accuracy']:.2f}%")
    print(f"  Unknown words accuracy: {stats['unknown_accuracy']:.2f}%")
    
    print("\n  Generating all standard visualizations...")
    create_all_visualizations(best_model, test_data, output_dir)
    
    # Summary
    print("\n" + "="*70)
    print("VISUALIZATION SUMMARY")
    print("="*70)
    print(f"\nAll visualizations saved to '{output_dir}/' directory:")
    print("  1. model_comparison.png - All model configurations")
    print("  2. models_word_types_comparison.png - Known vs unknown word performance")
    print("  3. suffix_patterns.png - Top 30 suffix patterns learned")
    print("  4. unknown_word_performance.png - Known vs unknown word accuracy")
    print("  5. transition_graph.png - State transition network")
    print("  6. transition_matrix.png - Transition probability heatmap")
    print("  7. initial_probabilities.png - Initial state probabilities")
    print("  8. confusion_matrix.png - Prediction confusion matrix")
    print("  9. viterbi_path_sample.png - Sample Viterbi decoding path")
    print("\n" + "="*70)


if __name__ == "__main__":
    main()