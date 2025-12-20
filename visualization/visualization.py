"""
Visualization utilities for HMM POS tagging.

This module provides functions to create various visualizations for analyzing
Hidden Markov Models including transition graphs, confusion matrices, emission
probabilities, and performance comparisons.
"""

import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import FancyBboxPatch
import networkx as nx


# =============================================================================
# CONSTANTS
# =============================================================================

# Standard professional color palette (matplotlib defaults)
PALETTE = {
    'primary': '#1f77b4',      # Blue
    'secondary': '#ff7f0e',    # Orange
    'accent1': '#2ca02c',      # Green
    'accent2': '#d62728',      # Red
    'accent3': '#9467bd',      # Purple
    'accent4': '#8c564b',      # Brown
    'neutral': '#7f7f7f',      # Gray
    'dark': '#000000',         # Black
}

DEFAULT_DPI = 300
DEFALT_COLORMAP = 'Blues'  # Standard blue colormap for all heatmaps (consistency)
WORD_TYPE_COLORS = {'known': PALETTE['accent1'], 'unknown': PALETTE['secondary']}


# =============================================================================
# GRAPH AND NETWORK VISUALIZATIONS
# =============================================================================


def visualize_transition_graph(hmm, min_prob=0.05, figsize=(16, 12), save_path=None):
    """Visualize HMM state transitions as a directed graph."""
    # Create directed graph
    G = nx.DiGraph()

    # Add nodes for each POS tag
    for tag in hmm.states:
        G.add_node(tag)

    # Add edges for significant transitions
    edge_labels = {}
    for i, tag_i in enumerate(hmm.states):
        for j, tag_j in enumerate(hmm.states):
            prob = hmm.transition_probs[i][j]
            if prob > min_prob:
                G.add_edge(tag_i, tag_j, weight=prob)
                edge_labels[(tag_i, tag_j)] = f"{prob:.3f}"

    # Create visualization
    plt.figure(figsize=figsize)
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

    # Draw nodes (size based on initial probability)
    node_sizes = [
        hmm.initial_probs[hmm.tag_to_idx[tag]] * 10000
        for tag in G.nodes()
    ]
    nx.draw_networkx_nodes(
        G, pos, node_size=node_sizes,
        node_color=PALETTE['secondary'], alpha=0.9,
        edgecolors=PALETTE['dark'], linewidths=2
    )

    # Draw node labels
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')

    # Draw edges with varying thickness based on probability
    edges = G.edges()
    weights = [G[u][v]['weight'] for u, v in edges]

    # Normalize weights for edge width
    max_weight = max(weights) if weights else 1
    edge_widths = [w / max_weight * 3 for w in weights]

    nx.draw_networkx_edges(G, pos, width=edge_widths,
                          alpha=0.5, edge_color=PALETTE['accent2'],
                          arrows=True, arrowsize=20,
                          arrowstyle='->', connectionstyle='arc3,rad=0.1')

    # Draw edge labels (transition probabilities)
    nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=7)

    plt.title(f"HMM State Transition Graph\n(Showing transitions with probability > {min_prob})",
              fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Transition graph saved to {save_path}")
    plt.close()


# =============================================================================
# MATRIX VISUALIZATIONS
# =============================================================================

def visualize_transition_matrix(hmm, figsize=(14, 12), save_path=None):
    """Visualize transition probability matrix as a heatmap."""
    # Use all tags
    all_tags = hmm.states

    # Create heatmap
    plt.figure(figsize=figsize)
    sns.heatmap(hmm.transition_probs, annot=True, fmt='.3f', cmap='Blues',
                xticklabels=all_tags, yticklabels=all_tags,
                cbar_kws={'label': 'Transition Probability'},
                linewidths=0.5, linecolor='white')

    plt.title(f'Transition Probability Matrix\nAll {len(all_tags)} POS Tags',
              fontsize=14, fontweight='bold')
    plt.xlabel('Next Tag', fontsize=12)
    plt.ylabel('Current Tag', fontsize=12)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=DEFAULT_DPI, bbox_inches='tight')
        print(f"Transition matrix saved to {save_path}")
    plt.close()


def visualize_confusion_matrix(hmm, test_data, figsize=(14, 12), save_path=None):
    """Compute and visualize confusion matrix for all POS tag states."""
    from evaluation import compute_confusion_matrix

    confusion_matrix = compute_confusion_matrix(hmm, test_data)
    all_tags = hmm.states

    # Normalize by row (true labels) to get percentages
    cm_normalized = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
    cm_normalized = np.nan_to_num(cm_normalized)  # Handle division by zero

    plt.figure(figsize=figsize)
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=all_tags, yticklabels=all_tags,
                cbar_kws={'label': 'Proportion'},
                linewidths=0.5, linecolor='white')

    plt.title(f'Confusion Matrix',
              fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Tag', fontsize=12)
    plt.ylabel('True Tag', fontsize=12)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=DEFAULT_DPI, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    plt.close()


# =============================================================================
# PROBABILITY VISUALIZATIONS
# =============================================================================

def visualize_emission_probabilities(hmm, tag='NOUN', top_n=20, figsize=(12, 6), save_path=None):
    """Visualize top emission probabilities for a specific POS tag."""
    if tag not in hmm.tag_to_idx:
        print(f"Tag '{tag}' not found. Available tags: {hmm.states}")
        return

    tag_idx = hmm.tag_to_idx[tag]
    emission_probs = hmm.emission_probs[tag_idx]

    # Get top N words
    top_indices = np.argsort(emission_probs)[::-1][:top_n]
    top_words = [hmm.idx_to_word[i] for i in top_indices]
    top_probs = emission_probs[top_indices]

    # Create bar plot
    plt.figure(figsize=figsize)
    bars = plt.barh(range(len(top_words)), top_probs, color='steelblue', alpha=0.8)

    # Color the highest probability bar differently
    bars[0].set_color('orange')

    plt.yticks(range(len(top_words)), top_words, fontsize=10)
    plt.xlabel('Emission Probability', fontsize=12)
    plt.title(f'Top {top_n} Words for POS Tag: {tag}', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()  # Highest at top

    # Add probability values on bars
    for i, (word, prob) in enumerate(zip(top_words, top_probs)):
        plt.text(prob + 0.0001, i, f'{prob:.4f}', va='center', fontsize=8)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=DEFAULT_DPI, bbox_inches='tight')
        print(f"Emission probabilities saved to {save_path}")
    plt.close()


def visualize_initial_probabilities(hmm, figsize=(12, 6), save_path=None):
    """Visualize initial state probabilities."""
    # Sort by probability
    sorted_indices = np.argsort(hmm.initial_probs)[::-1]
    sorted_tags = [hmm.states[i] for i in sorted_indices]
    sorted_probs = hmm.initial_probs[sorted_indices]

    plt.figure(figsize=figsize)
    bars = plt.bar(range(len(sorted_tags)), sorted_probs, color=PALETTE['primary'], alpha=0.8,
                   edgecolor=PALETTE['dark'], linewidth=1.2)

    # # Highlight top 3
    # for i in range(min(3, len(bars))):
    #     bars[i].set_color(PALETTE['accent1'])

    plt.xticks(range(len(sorted_tags)), sorted_tags, rotation=45, ha='right')
    plt.ylabel('Initial Probability (π)', fontsize=12)
    plt.title('Initial State Probabilities\nP(sentence starts with tag)',
              fontsize=14, fontweight='bold')
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Initial probabilities saved to {save_path}")
    plt.close()


def visualize_marginal_probabilities(hmm, train_data, figsize=(12, 6), save_path=None):
    """Visualize marginal probabilities (tag distribution in training data)."""
    # Count tag frequencies in training data
    tag_counts = {tag: 0 for tag in hmm.states}
    total_tags = 0

    for sentence in train_data:
        for word, tag in sentence:
            if tag in tag_counts:
                tag_counts[tag] += 1
                total_tags += 1

    # Convert to probabilities
    tag_probs = {tag: count / total_tags for tag, count in tag_counts.items()}

    # Sort by probability
    sorted_items = sorted(tag_probs.items(), key=lambda x: x[1], reverse=True)
    sorted_tags = [item[0] for item in sorted_items]
    sorted_probs = [item[1] for item in sorted_items]

    plt.figure(figsize=figsize)
    bars = plt.bar(range(len(sorted_tags)), sorted_probs, color=PALETTE['primary'], alpha=0.8,
                   edgecolor=PALETTE['dark'], linewidth=1.2)

    # # Highlight top 3
    # for i in range(min(3, len(bars))):
    #     bars[i].set_color(PALETTE['accent1'])

    plt.xticks(range(len(sorted_tags)), sorted_tags, rotation=45, ha='right')
    plt.ylabel('Marginal Probability', fontsize=12)
    plt.title('Tag Distribution in Training Data\nMarginal Probabilities P(tag)',
              fontsize=14, fontweight='bold')
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Marginal probabilities saved to {save_path}")
    plt.close()


def visualize_viterbi_path(hmm, sentence_words, true_tags=None, figsize=(16, 9), save_path=None):
    """Visualize the Viterbi path for a specific sentence."""
    predicted_tags = hmm.predict(sentence_words)

    fig, ax = plt.subplots(figsize=figsize)

    # Color scheme
    WORD_COLOR = '#E8F4F8'      # Soft light blue - for words
    CORRECT_COLOR = '#D4EDDA'    # Soft green - correct predictions
    WRONG_COLOR = '#F8D7DA'      # Soft red - wrong predictions
    ARROW_COLOR = '#6C757D'      # Gray - transition arrows
    EMISSION_COLOR = '#7952B3'   # Muted purple - emission arrows
    BORDER_COLOR = '#495057'     # Dark gray - borders

    # Calculate positions
    n_words = len(sentence_words)
    x_positions = np.arange(n_words)

    # Draw word boxes (bottom row)
    for i, word in enumerate(sentence_words):
        box = FancyBboxPatch((i - 0.4, -0.5), 0.8, 0.8,
                            boxstyle="round,pad=0.1",
                            edgecolor=BORDER_COLOR, facecolor=WORD_COLOR,
                            linewidth=2.5, alpha=0.9)
        ax.add_patch(box)
        # Truncate long words and display
        display_word = word[:12] + '...' if len(word) > 12 else word
        ax.text(i, -0.1, display_word, ha='center', va='center',
               fontsize=10, fontweight='bold', color=BORDER_COLOR)

    # Draw predicted tag boxes (top row) with color coding
    for i, tag in enumerate(predicted_tags):
        if true_tags and true_tags[i] != tag:
            color = WRONG_COLOR  # Wrong prediction
        else:
            color = CORRECT_COLOR  # Correct prediction

        box = FancyBboxPatch((i - 0.4, 1.2), 0.8, 0.8,
                            boxstyle="round,pad=0.1",
                            edgecolor=BORDER_COLOR, facecolor=color,
                            linewidth=2.5, alpha=0.9)
        ax.add_patch(box)
        ax.text(i, 1.6, tag, ha='center', va='center',
               fontsize=11, fontweight='bold', color=BORDER_COLOR)

    # Draw emission arrows (word to tag)
    for i in range(n_words):
        ax.annotate('', xy=(i, 1.2), xytext=(i, 0.3),
                   arrowprops=dict(arrowstyle='->', lw=2.5, color=EMISSION_COLOR, alpha=0.7))

    # Draw transition arrows between tags
    for i in range(n_words - 1):
        ax.annotate('', xy=(i + 1, 1.6), xytext=(i + 0.4, 1.6),
                   arrowprops=dict(arrowstyle='->', lw=2, color=ARROW_COLOR,
                                 alpha=0.6, linestyle='--'))

    # Draw true tags if provided (above predicted tags)
    if true_tags:
        for i, tag in enumerate(true_tags):
            # Show only if different from predicted
            if tag != predicted_tags[i]:
                ax.text(i, 2.5, f"True: {tag}", ha='center', va='center',
                       fontsize=9, style='italic', color='#DC143C',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFF0F0',
                                edgecolor='#DC143C', linewidth=1.5, alpha=0.8))

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=WORD_COLOR, edgecolor=BORDER_COLOR, label='Input Words', linewidth=2),
        Patch(facecolor=CORRECT_COLOR, edgecolor=BORDER_COLOR, label='Correct Predictions', linewidth=2),
        Patch(facecolor=WRONG_COLOR, edgecolor=BORDER_COLOR, label='Wrong Predictions', linewidth=2),
        Patch(facecolor='none', edgecolor=EMISSION_COLOR, label='Emission (Word→Tag)', linewidth=2),
        Patch(facecolor='none', edgecolor=ARROW_COLOR, label='Transition (Tag→Tag)', linewidth=2, linestyle='--')
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10, framealpha=0.95,
             edgecolor=BORDER_COLOR, fancybox=True)

    ax.set_xlim(-0.5, n_words - 0.5)
    ax.set_ylim(-1.2, 3.2)
    ax.axis('off')

    # Title with accuracy
    title = 'Viterbi Decoding Path Visualization'
    if true_tags:
        n_correct = sum(1 for p, t in zip(predicted_tags, true_tags) if p == t)
        accuracy = n_correct / len(true_tags) * 100
        title += f'\nAccuracy: {n_correct}/{len(true_tags)} ({accuracy:.1f}%)'

    plt.title(title, fontsize=16, fontweight='bold', pad=20, color=BORDER_COLOR)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Viterbi path saved to {save_path}")
    plt.close()


def visualize_model_comparison(models_dict, figsize=(12, 7), save_path=None):
    """Compare multiple HMM models with a bar chart."""
    plt.figure(figsize=figsize)

    model_names = list(models_dict.keys())
    accuracies = [models_dict[name] * 100 for name in model_names]

    colors = ['#5D6D7E', '#566573', '#4A5568', '#34495E', '#2C3E50', '#1C2833', '#17202A'][:len(model_names)]
    bars = plt.bar(model_names, accuracies, color=colors, alpha=0.85, edgecolor=PALETTE['dark'], linewidth=1.5)

    # Add value labels on top of bars
    for i, (bar, acc) in enumerate(zip(bars, accuracies)):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{acc:.2f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.ylabel('Accuracy (%)', fontsize=13, fontweight='bold')
    plt.xlabel('Model', fontsize=13, fontweight='bold')
    plt.title('HMM Model Comparison', fontsize=15, fontweight='bold', pad=20)
    plt.ylim(0, 100)
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")
    plt.close()


def visualize_suffix_patterns(hmm, top_n=30, figsize=(14, 10), save_path=None):
    """Visualize the most common suffix patterns learned by the suffix model."""
    if not hasattr(hmm, 'suffix_probs') or not hmm.suffix_probs:
        print("No suffix patterns found. Model must be trained with use_suffix_model=True.")
        return

    # Get suffix frequencies
    suffix_counts = {}
    for suffix in hmm.suffix_probs:
        suffix_counts[suffix] = sum(hmm.suffix_probs[suffix].values())

    # Sort by frequency
    sorted_suffixes = sorted(suffix_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]

    # Prepare data for heatmap
    suffixes = [s for s, _ in sorted_suffixes]
    tags = sorted(hmm.states)

    # Create matrix: rows=suffixes, cols=tags
    matrix = np.zeros((len(suffixes), len(tags)))
    for i, suffix in enumerate(suffixes):
        for j, tag in enumerate(tags):
            if tag in hmm.suffix_probs[suffix]:
                matrix[i][j] = hmm.suffix_probs[suffix][tag]

    # Create heatmap
    plt.figure(figsize=figsize)
    sns.heatmap(matrix, xticklabels=tags, yticklabels=suffixes,
               cmap='Blues', cbar_kws={'label': 'Probability'},
               linewidths=0.5, linecolor='white', annot=False)

    plt.xlabel('POS Tag', fontsize=12, fontweight='bold')
    plt.ylabel('Suffix Pattern', fontsize=12, fontweight='bold')
    plt.title(f'Top {top_n} Suffix Patterns and Their Tag Probabilities',
             fontsize=14, fontweight='bold', pad=15)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")
    plt.close()


def visualize_unknown_word_performance(hmm, test_data, method='viterbi', figsize=(10, 6), save_path=None):
    """Compare model performance on known vs unknown words."""
    # Separate predictions for known and unknown words
    known_correct = 0
    known_total = 0
    unknown_correct = 0
    unknown_total = 0

    for sentence in test_data:
        words = [word for word, _ in sentence]
        true_tags = [tag for _, tag in sentence]
        predicted_tags = hmm.predict(words, method=method)

        for word, true_tag, pred_tag in zip(words, true_tags, predicted_tags):
            if word in hmm.word_to_idx:
                known_total += 1
                if pred_tag == true_tag:
                    known_correct += 1
            else:
                unknown_total += 1
                if pred_tag == true_tag:
                    unknown_correct += 1

    # Calculate accuracies
    known_acc = (known_correct / known_total * 100) if known_total > 0 else 0
    unknown_acc = (unknown_correct / unknown_total * 100) if unknown_total > 0 else 0

    # Create bar chart
    plt.figure(figsize=figsize)

    categories = ['Known Words', 'Unknown Words']
    accuracies = [known_acc, unknown_acc]
    counts = [f'{known_correct}/{known_total}', f'{unknown_correct}/{unknown_total}']

    colors = ['#2ecc71', '#e74c3c']
    bars = plt.bar(categories, accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

    # Add value labels
    for bar, acc, count in zip(bars, accuracies, counts):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{acc:.2f}%\n({count})', ha='center', va='bottom',
                fontsize=11, fontweight='bold')

    plt.ylabel('Accuracy (%)', fontsize=13, fontweight='bold')
    plt.xlabel('Word Type', fontsize=13, fontweight='bold')
    plt.title('Model Performance: Known vs Unknown Words', fontsize=15, fontweight='bold', pad=20)
    plt.ylim(0, 100)
    plt.grid(axis='y', alpha=0.3, linestyle='--')

    # Add text annotation for model info
    model_params = f'Model: {hmm.states[0] if hasattr(hmm, "states") else "HMM"}'
    if hasattr(hmm, 'use_suffix_model') and hmm.use_suffix_model:
        model_params = f'Suffix model (n={hmm.suffix_length})'
    if hasattr(hmm, 'use_prefix_model') and hmm.use_prefix_model:
        if hasattr(hmm, 'use_suffix_model') and hmm.use_suffix_model:
            model_params = f'Prefix+Suffix (p={hmm.prefix_length}, s={hmm.suffix_length})'
        else:
            model_params = f'Prefix model (n={hmm.prefix_length})'

    plt.text(0.98, 0.02, f'{model_params}\\nTrained on training set',
             transform=plt.gca().transAxes,
             fontsize=9, style='italic', verticalalignment='bottom',
             horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor=PALETTE['neutral'], alpha=0.3, edgecolor=PALETTE['dark']))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")
    plt.close()

    return {
        'known_accuracy': known_acc,
        'unknown_accuracy': unknown_acc,
        'known_correct': known_correct,
        'known_total': known_total,
        'unknown_correct': unknown_correct,
        'unknown_total': unknown_total
    }


def compare_models_on_word_types(models_dict, test_data, method='viterbi', figsize=(12, 6), save_path=None):
    """Compare multiple models on known vs unknown words."""
    results = {}

    # Evaluate each model
    for model_name, hmm in models_dict.items():
        known_correct = 0
        known_total = 0
        unknown_correct = 0
        unknown_total = 0

        for sentence in test_data:
            words = [word for word, _ in sentence]
            true_tags = [tag for _, tag in sentence]
            predicted_tags = hmm.predict(words, method=method)

            for word, true_tag, pred_tag in zip(words, true_tags, predicted_tags):
                if word in hmm.word_to_idx:
                    known_total += 1
                    if pred_tag == true_tag:
                        known_correct += 1
                else:
                    unknown_total += 1
                    if pred_tag == true_tag:
                        unknown_correct += 1

        known_acc = (known_correct / known_total * 100) if known_total > 0 else 0
        unknown_acc = (unknown_correct / unknown_total * 100) if unknown_total > 0 else 0

        results[model_name] = {
            'known': known_acc,
            'unknown': unknown_acc
        }

    # Create grouped bar chart
    fig, ax = plt.subplots(figsize=figsize)

    model_names = list(results.keys())
    known_accs = [results[name]['known'] for name in model_names]
    unknown_accs = [results[name]['unknown'] for name in model_names]

    x = np.arange(len(model_names))
    width = 0.35

    bars1 = ax.bar(x - width/2, known_accs, width, label='Known Words',
                   color=PALETTE['accent4'], alpha=0.85, edgecolor=PALETTE['dark'], linewidth=1.5)
    bars2 = ax.bar(x + width/2, unknown_accs, width, label='Unknown Words',
                   color=PALETTE['accent3'], alpha=0.85, edgecolor=PALETTE['dark'], linewidth=1.5)

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                   f'{height:.1f}%', ha='center', va='bottom',
                   fontsize=10, fontweight='bold')

    ax.set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold')
    ax.set_xlabel('Model', fontsize=13, fontweight='bold')
    ax.set_title('Model Comparison: Known vs Unknown Words', fontsize=15, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.legend(fontsize=11, loc='lower right')
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")
    plt.close()

    return results


def compare_viterbi_vs_posterior(hmm, test_data, figsize=(12, 6), save_path=None):
    """Compare Viterbi and Posterior decoding performance."""
    # Evaluate both methods
    viterbi_results = hmm.evaluate(test_data, method='viterbi')
    posterior_results = hmm.evaluate(test_data, method='posterior')

    # Create comparison plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Plot 1: Accuracy comparison
    methods = ['Viterbi', 'Posterior']
    accuracies = [viterbi_results['accuracy'] * 100, posterior_results['accuracy'] * 100]
    colors = ['#3498db', '#e74c3c']

    bars = ax1.bar(methods, accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{acc:.2f}%', ha='center', va='bottom',
                fontsize=12, fontweight='bold')

    ax1.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Decoding Method', fontsize=12, fontweight='bold')
    ax1.set_title('Accuracy Comparison', fontsize=13, fontweight='bold')
    ax1.set_ylim(0, 100)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')

    # Plot 2: Correct vs Incorrect predictions
    x = np.arange(len(methods))
    width = 0.35

    correct = [viterbi_results['correct'], posterior_results['correct']]
    incorrect = [viterbi_results['incorrect'], posterior_results['incorrect']]

    bars1 = ax2.bar(x - width/2, correct, width, label='Correct',
                    color=PALETTE['accent4'], alpha=0.8, edgecolor=PALETTE['dark'], linewidth=1.5)
    bars2 = ax2.bar(x + width/2, incorrect, width, label='Incorrect',
                    color=PALETTE['accent1'], alpha=0.8, edgecolor=PALETTE['dark'], linewidth=1.5)

    ax2.set_ylabel('Number of Predictions', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Decoding Method', fontsize=12, fontweight='bold')
    ax2.set_title('Prediction Breakdown', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods)
    ax2.legend(fontsize=10)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')

    plt.suptitle('Viterbi vs Posterior Decoding Comparison',
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")
    plt.close()

    return {
        'viterbi': viterbi_results,
        'posterior': posterior_results
    }


def create_all_visualizations(hmm, test_data, output_dir='visualizations'):
    """Create all visualizations and save them."""
    import os
    os.makedirs(output_dir, exist_ok=True)

    print("\nGenerating all visualizations...")

    # 1. Transition graph
    print("1. Creating transition graph...")
    visualize_transition_graph(hmm, min_prob=0.05,
                              save_path=f'{output_dir}/transition_graph.png')

    # 2. Transition matrix heatmap (all states)
    print("2. Creating transition matrix heatmap...")
    visualize_transition_matrix(hmm,
                               save_path=f'{output_dir}/transition_matrix.png')

    # 3. Initial probabilities
    print("3. Creating initial probabilities plot...")
    visualize_initial_probabilities(hmm,
                                    save_path=f'{output_dir}/initial_probabilities.png')

    # 4. Confusion matrix (all states)
    print("4. Creating confusion matrix...")
    visualize_confusion_matrix(hmm, test_data,
                              save_path=f'{output_dir}/confusion_matrix.png')

    # 5. Sample Viterbi path
    print("5. Creating sample Viterbi path...")
    if test_data:
        sample = test_data[0]
        words = [w for w, t in sample]
        words = ['Ես',   'իմ',  'անուշ', 'Հայաստանի', 'արևահամ', 'բարն', 'եմ',  'սիրում', '։']
        true_tags =     ['PRON', 'DET', 'ADJ',   'PROPN',     'ADJ',     'NOUN', 'AUX', 'VERB',    'PUNCT']
        visualize_viterbi_path(hmm, words, true_tags,
                             save_path=f'{output_dir}/viterbi_path_sample.png')

    # 6. Viterbi vs Posterior comparison
    print("6. Comparing Viterbi vs Posterior decoding...")
    compare_viterbi_vs_posterior(hmm, test_data,
                                save_path=f'{output_dir}/viterbi_vs_posterior.png')

    print(f"\nAll visualizations saved to '{output_dir}/' directory!")
