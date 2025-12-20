# Hidden Markov Model - POS Tagger for Armenian

A comprehensive implementation of Hidden Markov Models (HMM) for Part-of-Speech (POS) tagging using the Armenian Universal Dependencies dataset. Features multiple HMM variants including bigram (first-order) and trigram (second-order) models with advanced unknown word handling strategies.

## Project Structure

```
hidden_markov_model_pos_tagging/
├── dataset/                     # CoNLL-U format datasets
│   ├── hy_armtdp-ud-train.conllu
│   ├── hy_armtdp-ud-dev.conllu
│   └── hy_armtdp-ud-test.conllu
├── src/                         # Core source code
│   ├── hmm.py                   # Main HMM implementation
│   ├── hmm_trigram.py           # Trigram (2nd-order) HMM
│   ├── data_utils.py            # Data loading utilities
│   ├── evaluation.py            # Evaluation and analysis
│   ├── config_loader.py         # Configuration loader
│   └── main.py                  # Main training script
├── examples/                    # Example scripts
│   ├── basic_usage.py           # Simple usage example
│   ├── compare_models.py        # Compare HMM variants
│   └── tag_custom_text.py       # Tag custom Armenian text
├── visualization/               # Visualization module
│   ├── visualization.py         # Visualization library
│   └── visualize_all.py         # Generate all visualizations
├── visualizations/              # Generated plots (created at runtime)
├── models/                      # Saved trained models
├── config.cfg                   # Configuration file
├── requirements.txt             # Python dependencies
├── DATASET.md                   # Dataset documentation
├── README.md                    # README file
└── HMM_for_POS.pdf              # Project report
```

## Quick Start

### 1. Installation

```bash
# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Train and Evaluate HMM

```bash
# Main training pipeline
python src/main.py

# Compare different HMM variants
python examples/compare_models.py

# Tag custom Armenian text
python examples/tag_custom_text.py
```

### 3. Generate Visualizations

```bash
# Generate all visualizations
python visualization/visualize_all.py

# Output will be saved to visualizations/ directory
```

This creates visualizations for model comparison, transition matrices, confusion matrices, and more.

## Features

✅ **Multiple HMM Variants**
- **Bigram HMM** (first-order): Classical HMM implementation
- **Trigram HMM** (second-order): Extended HMM for better context modeling
- **Suffix-enhanced models**: Improved unknown word handling using word endings
- **Prefix-enhanced models**: Unknown word handling using word beginnings
- **Prefix+Suffix combined**: Best performance by combining both strategies

✅ **Advanced Unknown Word Handling**
- Morphological analysis using prefix/suffix patterns
- Learned affix-to-tag probability distributions
- Fallback to marginal tag probabilities
- Achieves 92.06% accuracy on test set (best configuration)

✅ **Complete HMM Implementation**
- Object-oriented design with clean API
- Maximum Likelihood Estimation for parameters
- Viterbi algorithm (most likely sequence)
- Posterior decoding (marginal probabilities)
- Add-epsilon smoothing for unseen events

✅ **Data Processing**
- CoNLL-U file parsing
- Armenian Universal Dependencies dataset support
- Random 80-10-10 train/dev/test split (reproducible with seed)
- Lowercase normalization for vocabulary reduction

✅ **Evaluation & Analysis**
- Accuracy metrics (overall, known words, unknown words)
- Confusion matrix generation
- Per-tag performance statistics
- Model comparison across configurations

✅ **Visualization & Model Persistence**
- Comprehensive visualizations for model analysis
- Save/load trained models (pickle serialization)

## HMM Model Types

### Bigram HMM (First-Order)
Standard HMM where each tag depends only on the previous tag. Available variants:

1. **Classical HMM**: Basic implementation without special unknown word handling (86.40% accuracy)
2. **Suffix-enhanced**: Uses word endings for unknown word prediction (91-92% accuracy)
3. **Prefix-enhanced**: Uses word beginnings for unknown word prediction (~88.68% accuracy)
4. **Prefix+Suffix**: Combines both strategies (91-91.4% accuracy)

### Trigram HMM (Second-Order)
Extended HMM where each tag depends on the previous two tags. Due to increased sparsity, achieves 71.10% accuracy (lower than bigram models).

## Performance

Performance on Armenian UD dataset (test set accuracy):

| Model Type | Configuration                  | Accuracy   |
|------------|--------------------------------|------------|
| Bigram     | Classical (no affix)           | 86.40%     |
| Bigram     | Suffix-only (n=2)              | 91.76%     |
| Bigram     | Suffix-only (n=3)              | **92.06%** |
| Bigram     | Prefix-only (n=2)              | 88.68%     |
| Bigram     | Prefix-only (n=3)              | 88.68%     |
| Bigram     | Prefix+Suffix (pref=3, suff=2) | 91.12%     |
| Bigram     | Prefix+Suffix (pref=2, suff=3) | 91.39%     |
| Bigram     | Prefix+Suffix (pref=3, suff=3) | 91.29%     |
| Trigram    | Second-order HMM               | 71.10%     |

**Note**: The main.py script automatically trains multiple configurations and selects the best model based on development set performance.

## HMM Components

Each model learns four key probability distributions:

1. **States (Q)** - Set of POS tags
   - Example: {NOUN, VERB, ADJ, ADP, PUNCT, DET, ...}
   - 17 universal POS tags in Armenian UD dataset

2. **Initial Probabilities (π)** - Starting tag probabilities
   - P(first tag in sentence)
   - Bigram: single tag probabilities
   - Trigram: tag pair probabilities

3. **Transition Probabilities (A)** - Tag sequence probabilities
   - Bigram: P(tag_j | tag_i)
   - Trigram: P(tag_k | tag_i, tag_j)

4. **Emission Probabilities (B)** - Word-tag associations
   - P(word | tag) for known words
   - P(tag | suffix/prefix) for unknown words (enhanced models)

## File Descriptions

### Core Files
- **`config.cfg`** - Configuration file for dataset paths and output settings
- **`src/hmm.py`** - HiddenMarkovModel class with train/predict/evaluate methods
- **`src/hmm_trigram.py`** - TrigramHMM class for second-order HMM
- **`src/data_utils.py`** - Functions for loading and processing CoNLL-U datasets
- **`src/evaluation.py`** - Evaluation metrics, error analysis, and reporting
- **`src/config_loader.py`** - Utility to load and parse configuration file
- **`src/main.py`** - Main training and evaluation pipeline

### Visualization
- **`visualization/visualization.py`** - Visualization functions for HMM analysis
- **`visualization/visualize_all.py`** - Generate comprehensive visualizations

### Examples
- **`examples/basic_usage.py`** - Simple usage example
- **`examples/compare_models.py`** - Compare different HMM variants
- **`examples/tag_custom_text.py`** - Tag custom Armenian text

## References

- Armenian UD Treebank: https://github.com/UniversalDependencies/UD_Armenian-ArmTDP

## License

Educational project for ASDS-25 (YSU) Statistics course.

## Authors

- Mane Mkhitaryan [@ManeMkh] (https://github.com/ManeMkh)
- Eduard Danielyan [@DanielyanEduard] (https://github.com/DanielyanEduard)
- Albert Piliposyan [@albpiliposyan] (https://github.com/albpiliposyan)
