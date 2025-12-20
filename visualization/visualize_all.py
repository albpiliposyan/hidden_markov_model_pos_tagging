"""Generate all HMM POS tagging visualizations."""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import FancyBboxPatch, Patch
import networkx as nx
from hmm import HiddenMarkovModel
from hmm_trigram import TrigramHMM
from data_utils import load_armenian_dataset
from evaluation import compute_confusion_matrix
from visualization import (
    visualize_transition_graph,
    visualize_transition_matrix,
    visualize_confusion_matrix,
    visualize_initial_probabilities,
    visualize_marginal_probabilities,
    visualize_viterbi_path,
    visualize_suffix_patterns
)


# Config
OUTPUT_DIR = 'visualizations'
PALETTE = {
    'primary': '#1f77b4', 'secondary': '#ff7f0e', 'accent1': '#2ca02c',
    'accent2': '#d62728', 'accent3': '#9467bd', 'neutral': '#7f7f7f', 'dark': '#000000',
}
MODEL_COLORS = {
    'classical': PALETTE['neutral'], 'suffix': PALETTE['primary'],
    'prefix': PALETTE['secondary'], 'prefix_suffix': PALETTE['accent1'],
}


# Model configs
def get_configs():
    return [
        {'name': 'Bigram HMM', 'params': {'use_suffix_model': False, 'use_prefix_model': False}},
        {'name': 'Trigram HMM', 'params': {'model_type': 'trigram'}},
        {'name': 'Suf-2', 'params': {'use_suffix_model': True, 'suffix_length': 2, 'use_prefix_model': False}},
        {'name': 'Suf-3', 'params': {'use_suffix_model': True, 'suffix_length': 3, 'use_prefix_model': False}},
        {'name': 'Pref-2', 'params': {'use_suffix_model': False, 'use_prefix_model': True, 'prefix_length': 2}},
        {'name': 'Pref-3', 'params': {'use_suffix_model': False, 'use_prefix_model': True, 'prefix_length': 3}},
        {'name': 'P3+S2', 'params': {'use_suffix_model': True, 'suffix_length': 2, 'use_prefix_model': True, 'prefix_length': 3}},
        {'name': 'P2+S3', 'params': {'use_suffix_model': True, 'suffix_length': 3, 'use_prefix_model': True, 'prefix_length': 2}},
        {'name': 'P3+S3', 'params': {'use_suffix_model': True, 'suffix_length': 3, 'use_prefix_model': True, 'prefix_length': 3}},
    ]


def get_model_color(name):
    if 'Trigram HMM' in name: return PALETTE['accent3']
    if 'Bigram HMM' in name: return MODEL_COLORS['classical']
    if '+' in name: return MODEL_COLORS['prefix_suffix']
    if 'Suf' in name: return MODEL_COLORS['suffix']
    if 'Pref' in name: return MODEL_COLORS['prefix']
    return PALETTE['neutral']


def get_selected_models():
    return ['Bigram HMM', 'Pref-2', 'Pref-3', 'Suf-3', 'P3+S3', 'P2+S3']


# Train and evaluate
def train_model(train_data, dev_data, test_data, name, **params):
    print(f"  {name}...", end=' ')
    
    if params.pop('model_type', None) == 'trigram':
        model = TrigramHMM()
    else:
        model = HiddenMarkovModel(**params)
    
    model.train(train_data)
    dev_acc = model.evaluate(dev_data)['accuracy'] * 100
    test_acc = model.evaluate(test_data)['accuracy'] * 100
    print(f"Dev: {dev_acc:.2f}%, Test: {test_acc:.2f}%")
    
    return model, dev_acc, test_acc


# Utilities
def calculate_word_type_accuracy(model, test_data):
    """Calculate accuracy for known vs unknown words."""
    vocab = set(model.word_to_idx.keys())
    known_c = known_t = unk_c = unk_t = 0
    
    for sent in test_data:
        words = [w for w, _ in sent]
        true_tags = [t for _, t in sent]
        pred_tags = model.predict(words)
        
        for w, true_t, pred_t in zip(words, true_tags, pred_tags):
            if w.lower() in vocab:
                known_t += 1
                if true_t == pred_t: known_c += 1
            else:
                unk_t += 1
                if true_t == pred_t: unk_c += 1
    
    return {
        'known_acc': (known_c / known_t * 100) if known_t > 0 else 0,
        'unknown_acc': (unk_c / unk_t * 100) if unk_t > 0 else 0,
        'known_count': f'{known_c}/{known_t}',
        'unknown_count': f'{unk_c}/{unk_t}'
    }


# Visualizations
def plot_model_comparison(results, output_dir):
    """Bar chart: all models dev/test accuracy"""
    fig, ax = plt.subplots(figsize=(20, 10))
    
    names = list(results.keys())
    dev = [results[n]['dev'] for n in names]
    test = [results[n]['test'] for n in names]
    colors = [get_model_color(n) for n in names]
    
    x = np.arange(len(names))
    w = 0.35
    
    bars1 = ax.bar(x - w/2, dev, w, label='Validation', color=colors, edgecolor='black', linewidth=1.2, alpha=0.8)
    bars2 = ax.bar(x + w/2, test, w, label='Test', color=colors, edgecolor='black', linewidth=1.2, alpha=0.5)
    
    for bars, vals in [(bars1, dev), (bars2, test)]:
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15,
                   f'{val:.2f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # ax.set_xlabel('Model', fontsize=13, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold')
    # ax.set_title('Comparison', fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=11)
    ax.set_ylim([min(dev + test) - 2, max(dev + test) + 3])
    ax.grid(axis='y', alpha=0.3)
    ax.legend(fontsize=11, loc='upper left')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: model_comparison.png")


def plot_word_types(models, test_data, output_dir):
    """Bar chart: known vs unknown word accuracy"""
    results = {}
    
    for name, model in models.items():
        stats = calculate_word_type_accuracy(model, test_data)
        results[name] = {
            'known': stats['known_acc'],
            'unknown': stats['unknown_acc']
        }
    
    fig, ax = plt.subplots(figsize=(14, 8))
    names = list(results.keys())
    known = [results[n]['known'] for n in names]
    unknown = [results[n]['unknown'] for n in names]
    
    x = np.arange(len(names))
    w = 0.35
    
    ax.bar(x - w/2, known, w, label='Known', color=PALETTE['accent1'], edgecolor='black', linewidth=1.2, alpha=0.85)
    ax.bar(x + w/2, unknown, w, label='Unknown', color=PALETTE['accent2'], edgecolor='black', linewidth=1.2, alpha=0.85)
    
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Known vs Unknown Words', fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/models_word_types_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: models_word_types_comparison.png")


def plot_viterbi_vs_posterior(model, test_data, output_dir):
    """Bar chart: Viterbi vs Posterior accuracy"""
    vit = model.evaluate(test_data, method='viterbi')
    post = model.evaluate(test_data, method='posterior')
    
    fig, ax = plt.subplots(figsize=(10, 6))
    methods = ['Viterbi', 'Posterior']
    accs = [vit['accuracy'] * 100, post['accuracy'] * 100]
    colors = [PALETTE['primary'], PALETTE['secondary']]
    
    bars = ax.bar(methods, accs, color=colors, alpha=0.85, edgecolor='black', linewidth=1.2)
    
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
               f'{acc:.2f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Viterbi vs Posterior Decoding', fontsize=14, fontweight='bold', pad=20)
    ax.set_ylim([min(accs) - 2, max(accs) + 3])
    ax.grid(axis='y', alpha=0.3)
    
    # Info box
    diff = accs[0] - accs[1]
    info = f'Difference: {diff:.2f}%\nViterbi: {vit["correct"]}/{vit["total_tokens"]}\nPosterior: {post["correct"]}/{post["total_tokens"]}'
    ax.text(0.98, 0.02, info, transform=ax.transAxes, fontsize=10,
           verticalalignment='bottom', horizontalalignment='right',
           bbox=dict(boxstyle='round', facecolor=PALETTE['neutral'], alpha=0.2, edgecolor='black'))
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/viterbi_vs_posterior.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: viterbi_vs_posterior.png")


def plot_marginal_probabilities(model, train_data, output_dir):
    """Bar chart: tag distribution in training data"""
    tag_counts = {tag: 0 for tag in model.states}
    total = 0
    
    for sent in train_data:
        for _, tag in sent:
            if tag in tag_counts:
                tag_counts[tag] += 1
                total += 1
    
    items = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)
    tags = [t for t, _ in items]
    probs = [c / total for _, c in items]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(range(len(tags)), probs, color=PALETTE['primary'], alpha=0.8, edgecolor='black', linewidth=1.2)

    # # Highlight top 3    
    # for i in range(min(3, len(bars))):
    #     bars[i].set_color(PALETTE['accent1'])
    
    ax.set_xticks(range(len(tags)))
    ax.set_xticklabels(tags, rotation=45, ha='right')
    ax.set_ylabel('Probability', fontsize=12)
    # ax.set_title('Tag Distribution', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/marginal_probabilities.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: marginal_probabilities.png")


def plot_transition_graph(model, output_dir, min_prob=0.05):
    """Network graph: HMM state transitions"""
    visualize_transition_graph(model, min_prob=min_prob, save_path=f'{output_dir}/transition_graph.png')
    print(f"  Saved: transition_graph.png")


def plot_transition_matrix(model, output_dir):
    """Heatmap: transition probability matrix"""
    visualize_transition_matrix(model, save_path=f'{output_dir}/transition_matrix.png')
    print(f"  Saved: transition_matrix.png")


def plot_confusion_matrix(model, test_data, output_dir):
    """Heatmap: confusion matrix"""
    visualize_confusion_matrix(model, test_data, save_path=f'{output_dir}/confusion_matrix.png')
    print(f"  Saved: confusion_matrix.png")


def plot_initial_probabilities(model, output_dir):
    """Bar chart: initial state probabilities"""
    visualize_initial_probabilities(model, save_path=f'{output_dir}/initial_probabilities.png')
    print(f"  Saved: initial_probabilities.png")


def plot_viterbi_path(model, output_dir):
    """Path visualization: sample Viterbi decoding"""
    words =     [ 'Ես',  'իմ', 'անուշ', 'Հայաստանի', 'արևահամ', 'բառն', 'եմ', 'սիրում',  '։'  ]
    true_tags = ['PRON', 'DET', 'ADJ',    'PROPN',     'ADJ',   'NOUN', 'AUX', 'VERB', 'PUNCT']
    
    pred_tags = model.predict(words)
    
    fig, ax = plt.subplots(figsize=(16, 9))
    
    # Brighter, more vibrant colors
    WORD_COLOR = '#87CEEB'       # Sky blue - brighter for words
    CORRECT_COLOR = '#90EE90'    # Light green - brighter for correct predictions
    WRONG_COLOR = '#FFB6C1'      # Light pink - brighter for wrong predictions
    ARROW_COLOR = '#4169E1'      # Royal blue - brighter for transitions
    EMISSION_COLOR = '#9370DB'   # Medium purple - brighter for emissions
    BORDER_COLOR = '#2F4F4F'     # Dark slate gray - borders
    
    n_words = len(words)
    
    # Word boxes
    for i, word in enumerate(words):
        box = FancyBboxPatch((i - 0.4, -0.5), 0.8, 0.8, boxstyle="round,pad=0.1",
                            edgecolor=BORDER_COLOR, facecolor=WORD_COLOR,
                            linewidth=2.5, alpha=0.95)
        ax.add_patch(box)
        display_word = word[:12] + '...' if len(word) > 12 else word
        ax.text(i, -0.1, display_word, ha='center', va='center',
               fontsize=10, fontweight='bold', color=BORDER_COLOR)
    
    # Tag boxes
    for i, tag in enumerate(pred_tags):
        color = WRONG_COLOR if true_tags[i] != tag else CORRECT_COLOR
        box = FancyBboxPatch((i - 0.4, 1.2), 0.8, 0.8, boxstyle="round,pad=0.1",
                            edgecolor=BORDER_COLOR, facecolor=color,
                            linewidth=2.5, alpha=0.95)
        ax.add_patch(box)
        ax.text(i, 1.6, tag, ha='center', va='center',
               fontsize=11, fontweight='bold', color=BORDER_COLOR)
    
    # Emission arrows (word to tag)
    for i in range(n_words):
        ax.annotate('', xy=(i, 1.2), xytext=(i, 0.3),
                   arrowprops=dict(arrowstyle='->', lw=2.5, color=EMISSION_COLOR, alpha=0.8))
    
    # Transition arrows (tag to tag) - corrected to go between tag boxes
    for i in range(n_words - 1):
        ax.annotate('', xy=(i + 0.6, 1.6), xytext=(i + 0.4, 1.6),
                   arrowprops=dict(arrowstyle='->', lw=2.5, color=ARROW_COLOR,
                                 alpha=0.8, connectionstyle='arc3,rad=0'))
    
    # True tags (show only differences)
    for i, tag in enumerate(true_tags):
        if tag != pred_tags[i]:
            ax.text(i, 2.5, f"True: {tag}", ha='center', va='center',
                   fontsize=9, style='italic', color='#DC143C',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFF0F0',
                            edgecolor='#DC143C', linewidth=1.5, alpha=0.9))
    
    # Legend
    legend_elements = [
        Patch(facecolor=WORD_COLOR, edgecolor=BORDER_COLOR, label='Input Words', linewidth=2),
        Patch(facecolor=CORRECT_COLOR, edgecolor=BORDER_COLOR, label='Correct Predictions', linewidth=2),
        Patch(facecolor=WRONG_COLOR, edgecolor=BORDER_COLOR, label='Wrong Predictions', linewidth=2),
        Patch(facecolor='none', edgecolor=EMISSION_COLOR, label='Emission (Word→Tag)', linewidth=2),
        Patch(facecolor='none', edgecolor=ARROW_COLOR, label='Transition (Tag→Tag)', linewidth=2)
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10, framealpha=0.95,
             edgecolor=BORDER_COLOR, fancybox=True)
    
    ax.set_xlim(-0.5, n_words - 0.5)
    ax.set_ylim(-1.2, 3.2)
    ax.axis('off')
    
    n_correct = sum(1 for p, t in zip(pred_tags, true_tags) if p == t)
    accuracy = n_correct / len(true_tags) * 100
    title = f'Viterbi Decoding Path Visualization\nAccuracy: {n_correct}/{len(true_tags)} ({accuracy:.1f}%)'
    
    plt.title(title, fontsize=16, fontweight='bold', pad=20, color=BORDER_COLOR)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/viterbi_path_sample.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: viterbi_path_sample.png")


def plot_suffix_patterns(model, output_dir, top_n=30):
    """Heatmap: suffix patterns and their tag probabilities"""
    if not hasattr(model, 'suffix_probs') or not model.suffix_probs:
        print(f"  Skipped: suffix_patterns.png (no suffix model)")
        return
    visualize_suffix_patterns(model, top_n=top_n, save_path=f'{output_dir}/suffix_patterns.png')
    print(f"  Saved: suffix_patterns.png")


# Main
def main():
    print("="*70)
    print("GENERATING HMM POS TAGGING VISUALIZATIONS")
    print("="*70)
    
    # Load data
    print("\n[1] Loading dataset...")
    train_data, dev_data, test_data = load_armenian_dataset()
    print(f"  Train: {len(train_data)}, Dev: {len(dev_data)}, Test: {len(test_data)}")
    
    # Train models
    print("\n[2] Training models...")
    configs = get_configs()
    results = {}
    models = {}
    
    for cfg in configs:
        model, dev_acc, test_acc = train_model(train_data, dev_data, test_data, cfg['name'], **cfg['params'])
        results[cfg['name']] = {'dev': dev_acc, 'test': test_acc}
        models[cfg['name']] = model
    
    # Best model
    best = max(results.items(), key=lambda x: x[1]['test'])
    print(f"\n  Best: {best[0]} (Test: {best[1]['test']:.2f}%)")
    best_model = models[best[0]]
    
    # Create output dir
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Generate visualizations
    print("\n[3] Generating visualizations...")
    
    plot_model_comparison(results, OUTPUT_DIR)
    
    selected = {n: models[n] for n in get_selected_models()}
    plot_word_types(selected, test_data, OUTPUT_DIR)
    
    plot_viterbi_vs_posterior(best_model, test_data, OUTPUT_DIR)
    plot_marginal_probabilities(best_model, train_data, OUTPUT_DIR)
    plot_transition_graph(best_model, OUTPUT_DIR)
    plot_transition_matrix(best_model, OUTPUT_DIR)
    plot_confusion_matrix(best_model, test_data, OUTPUT_DIR)
    plot_initial_probabilities(best_model, OUTPUT_DIR)
    plot_viterbi_path(best_model, OUTPUT_DIR)
    plot_suffix_patterns(best_model, OUTPUT_DIR)
    
    print("\n" + "="*70)
    print("COMPLETE! Check visualizations/ directory")
    print("="*70)


if __name__ == "__main__":
    main()
