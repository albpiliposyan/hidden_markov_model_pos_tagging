"""Visualization utilities for HMM POS tagging."""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import FancyBboxPatch
import networkx as nx


def visualize_transition_graph(hmm, min_prob=0.05, figsize=(16, 12), save_path=None):
    """
    Visualize HMM state transitions as a directed graph.
    
    Args:
        hmm: Trained HMM model
        min_prob: Minimum transition probability to display (filters weak connections)
        figsize: Figure size
        save_path: Optional path to save the figure
    """
    # Create directed graph
    G = nx.DiGraph()
    
    # Add nodes (POS tags)
    for tag in hmm.Q:
        G.add_node(tag)
    
    # Add edges (transitions) with probabilities
    edge_labels = {}
    for i, tag_i in enumerate(hmm.Q):
        for j, tag_j in enumerate(hmm.Q):
            prob = hmm.A[i][j]
            if prob > min_prob:  # Only show significant transitions
                G.add_edge(tag_i, tag_j, weight=prob)
                edge_labels[(tag_i, tag_j)] = f"{prob:.3f}"
    
    # Create plot
    plt.figure(figsize=figsize)
    
    # Layout - circular or spring layout
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    # Alternative: pos = nx.circular_layout(G)
    
    # Draw nodes
    node_sizes = [hmm.pi[hmm.tag_to_idx[tag]] * 10000 for tag in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, 
                          node_color='lightblue', alpha=0.9,
                          edgecolors='black', linewidths=2)
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
    
    # Draw edges with varying thickness based on probability
    edges = G.edges()
    weights = [G[u][v]['weight'] for u, v in edges]
    
    # Normalize weights for edge width
    max_weight = max(weights) if weights else 1
    edge_widths = [w / max_weight * 3 for w in weights]
    
    nx.draw_networkx_edges(G, pos, width=edge_widths, 
                          alpha=0.6, edge_color='gray',
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
    
    plt.show()


def visualize_transition_matrix(hmm, figsize=(14, 12), save_path=None):
    """Visualize transition probability matrix as a heatmap for all states."""
    # Use all tags
    all_tags = hmm.Q
    
    # Create heatmap
    plt.figure(figsize=figsize)
    sns.heatmap(hmm.A, annot=True, fmt='.3f', cmap='YlOrRd',
                xticklabels=all_tags, yticklabels=all_tags,
                cbar_kws={'label': 'Transition Probability'},
                linewidths=0.5, linecolor='gray')
    
    plt.title(f'Transition Probability Matrix (A)\nAll {len(all_tags)} POS Tags', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Next Tag', fontsize=12)
    plt.ylabel('Current Tag', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Transition matrix saved to {save_path}")
    
    plt.show()


def visualize_emission_probabilities(hmm, tag='NOUN', top_n=20, figsize=(12, 6), save_path=None):
    """
    Visualize top emission probabilities for a specific POS tag.
    
    Args:
        hmm: Trained HMM model
        tag: POS tag to visualize
        top_n: Number of top words to show
        figsize: Figure size
        save_path: Optional path to save the figure
    """
    if tag not in hmm.tag_to_idx:
        print(f"Tag '{tag}' not found. Available tags: {hmm.Q}")
        return
    
    tag_idx = hmm.tag_to_idx[tag]
    emission_probs = hmm.B[tag_idx]
    
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
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Emission probabilities saved to {save_path}")
    
    plt.show()


def visualize_initial_probabilities(hmm, figsize=(12, 6), save_path=None):
    """
    Visualize initial state probabilities.
    
    Args:
        hmm: Trained HMM model
        figsize: Figure size
        save_path: Optional path to save the figure
    """
    # Sort by probability
    sorted_indices = np.argsort(hmm.pi)[::-1]
    sorted_tags = [hmm.Q[i] for i in sorted_indices]
    sorted_probs = hmm.pi[sorted_indices]
    
    plt.figure(figsize=figsize)
    bars = plt.bar(range(len(sorted_tags)), sorted_probs, color='teal', alpha=0.7)
    
    # Highlight top 3
    for i in range(min(3, len(bars))):
        bars[i].set_color('coral')
    
    plt.xticks(range(len(sorted_tags)), sorted_tags, rotation=45, ha='right')
    plt.ylabel('Initial Probability (π)', fontsize=12)
    plt.title('Initial State Probabilities\nP(sentence starts with tag)', 
              fontsize=14, fontweight='bold')
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Initial probabilities saved to {save_path}")
    
    plt.show()


def visualize_viterbi_path(hmm, sentence_words, true_tags=None, figsize=(14, 8), save_path=None):
    """
    Visualize the Viterbi path for a specific sentence.
    
    Args:
        hmm: Trained HMM model
        sentence_words: List of words
        true_tags: Optional list of true tags for comparison
        figsize: Figure size
        save_path: Optional path to save the figure
    """
    predicted_tags = hmm.predict(sentence_words)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Calculate positions
    n_words = len(sentence_words)
    x_positions = np.arange(n_words)
    
    # Draw word boxes
    for i, word in enumerate(sentence_words):
        # Word box
        box = FancyBboxPatch((i - 0.4, -0.5), 0.8, 0.8,
                            boxstyle="round,pad=0.1", 
                            edgecolor='black', facecolor='lightblue',
                            linewidth=2)
        ax.add_patch(box)
        ax.text(i, -0.1, word[:15], ha='center', va='center', 
               fontsize=9, fontweight='bold')
    
    # Draw predicted tag boxes
    for i, tag in enumerate(predicted_tags):
        color = 'lightgreen'
        if true_tags and true_tags[i] != tag:
            color = 'lightcoral'  # Wrong prediction
        
        box = FancyBboxPatch((i - 0.4, 1.2), 0.8, 0.8,
                            boxstyle="round,pad=0.1", 
                            edgecolor='black', facecolor=color,
                            linewidth=2)
        ax.add_patch(box)
        ax.text(i, 1.6, tag, ha='center', va='center', 
               fontsize=9, fontweight='bold')
    
    # Draw arrows showing the path
    for i in range(n_words):
        ax.arrow(i, 0.3, 0, 0.8, head_width=0.15, head_length=0.1, 
                fc='gray', ec='gray', alpha=0.6)
        
        if i < n_words - 1:
            # Transition arrows
            ax.arrow(i + 0.3, 1.6, 0.4, 0, head_width=0.1, head_length=0.1, 
                    fc='blue', ec='blue', alpha=0.4, linestyle='--')
    
    # Draw true tags if provided
    if true_tags:
        for i, tag in enumerate(true_tags):
            ax.text(i, 2.5, f"True: {tag}", ha='center', va='center', 
                   fontsize=8, style='italic', color='darkblue')
    
    ax.set_xlim(-0.5, n_words - 0.5)
    ax.set_ylim(-1, 3)
    ax.axis('off')
    
    title = 'Viterbi Path Visualization'
    if true_tags:
        n_correct = sum(1 for p, t in zip(predicted_tags, true_tags) if p == t)
        accuracy = n_correct / len(true_tags) * 100
        title += f'\nAccuracy: {accuracy:.1f}%'
    
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Viterbi path saved to {save_path}")
    
    plt.show()


def visualize_confusion_matrix(hmm, test_data, figsize=(14, 12), save_path=None):
    """Compute and visualize confusion matrix for all states."""
    from evaluation import compute_confusion_matrix
    
    confusion_matrix = compute_confusion_matrix(hmm, test_data)
    all_tags = hmm.Q
    
    # Normalize by row (true labels) to get percentages
    cm_normalized = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
    cm_normalized = np.nan_to_num(cm_normalized)  # Handle division by zero
    
    plt.figure(figsize=figsize)
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=all_tags, yticklabels=all_tags,
                cbar_kws={'label': 'Proportion'},
                linewidths=0.5, linecolor='white')
    
    plt.title(f'Confusion Matrix (Normalized)\nAll {len(all_tags)} POS Tags', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Tag', fontsize=12)
    plt.ylabel('True Tag', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    
    plt.show()


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
        true_tags = [t for w, t in sample]
        visualize_viterbi_path(hmm, words[:10], true_tags[:10],
                             save_path=f'{output_dir}/viterbi_path_sample.png')
    
    print(f"\nAll visualizations saved to '{output_dir}/' directory!")
