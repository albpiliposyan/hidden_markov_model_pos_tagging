# Visualization Guide

This document explains all visualizations generated for the HMM POS tagging project.

## How to Generate Visualizations

Run the visualization script:
```bash
python visualize_all.py
```

All visualizations will be saved to the `visualizations/` directory.

## Available Visualizations

### 1. Model Comparison (`model_comparison.png`)
**Purpose:** Compare accuracy of all three HMM variants
- **Basic HMM:** ~79.87% (uniform unknown word handling)
- **Suffix-Enhanced HMM:** ~89.68% (suffix-based unknown words)
- **Trigram HMM:** ~68.24% (second-order model)

**Use for:** Showing the improvement achieved by the suffix model

### 2. Suffix Patterns (`suffix_patterns.png`)
**Purpose:** Heatmap of top 30 suffix patterns and their tag probabilities
- Shows what the suffix model learned (e.g., "-ում" → VERB, "-ական" → ADJ)
- Rows: Most common 3-character suffixes
- Columns: POS tags
- Color intensity: Probability P(tag|suffix)

**Use for:** Understanding what patterns the model discovered

### 3. Unknown Word Performance (`unknown_word_performance.png`)
**Purpose:** Compare accuracy on known vs unknown words
- **Known words:** ~95.98% accuracy
- **Unknown words:** ~71.88% accuracy

**Use for:** Demonstrating model robustness on out-of-vocabulary words

### 4. Transition Graph (`transition_graph.png`)
**Purpose:** Network visualization of state transitions
- Nodes: POS tags (size = initial probability)
- Edges: Transitions with probability > 5%
- Shows which tags commonly follow others

**Use for:** Understanding tag sequence patterns

### 5. Transition Matrix (`transition_matrix.png`)
**Purpose:** Heatmap of full transition probability matrix A
- 17×17 matrix showing P(tag_j | tag_i) for all tag pairs
- Rows must sum to 1 (probability distribution)

**Use for:** Detailed view of all state transitions

### 6. Initial Probabilities (`initial_probabilities.png`)
**Purpose:** Bar chart of sentence-starting tag probabilities
- Shows which tags commonly start sentences (e.g., NOUN, PROPN, DET)

**Use for:** Understanding sentence structure patterns

### 7. Confusion Matrix (`confusion_matrix.png`)
**Purpose:** Shows which tags are confused with each other
- Rows: True tags
- Columns: Predicted tags
- Diagonal: Correct predictions
- Off-diagonal: Errors

**Use for:** Error analysis and identifying problematic tag pairs

### 8. Viterbi Path Sample (`viterbi_path_sample.png`)
**Purpose:** Visualize Viterbi decoding on a sample sentence
- Shows predicted vs true tag sequence
- Illustrates how the model tags a specific sentence

**Use for:** Demonstrating the tagging process

## Key Insights from Visualizations

1. **Suffix Model Impact:** The suffix-enhanced model shows +9.81 percentage points improvement over the basic model

2. **Known vs Unknown:** Model performs well on known words (95.98%) and reasonably on unknown words (71.88%)

3. **Common Transitions:** 
   - ADJ → NOUN (adjectives before nouns)
   - DET → NOUN (determiners before nouns)
   - NOUN → ADP (nouns before adpositions)

4. **Suffix Patterns:** Top suffixes clearly associated with specific tags (morphological information)

## Using Individual Visualization Functions

You can also create individual visualizations in your own scripts:

```python
from visualization.visualization import (
    visualize_model_comparison,
    visualize_suffix_patterns,
    visualize_unknown_word_performance
)

# Model comparison
models = {'Model A': 0.85, 'Model B': 0.90}
visualize_model_comparison(models, save_path='comparison.png')

# Suffix patterns
visualize_suffix_patterns(hmm, top_n=30, save_path='suffixes.png')

# Unknown word performance
stats = visualize_unknown_word_performance(hmm, test_data, save_path='unknown.png')
print(f"Unknown word accuracy: {stats['unknown_accuracy']:.2f}%")
```

## Statistics Summary

From the latest run:
- **Training data:** 2,223 sentences (train + dev combined)
- **Test data:** 277 sentences
- **Vocabulary size:** 12,986 unique words
- **Unknown words in test:** 26.14%
- **Suffix patterns learned:** 2,209 patterns

## Recommended Visualizations for Presentation

For academic papers or presentations, prioritize:
1. **Model Comparison** - Shows your contribution clearly
2. **Unknown Word Performance** - Demonstrates robustness
3. **Confusion Matrix** - For error analysis
4. **Suffix Patterns** - Shows what the model learned

All figures are saved at 300 DPI for publication quality.
