# Hidden Markov Model - POS Tagger for Armenian

A complete implementation of Hidden Markov Models for Part-of-Speech tagging using the Armenian Universal Dependencies dataset.

## Project Structure

```
hidden_markov_model/
├── dataset/                   # CoNLL-U format datasets
│   ├── hy_armtdp-ud-train.conllu
│   ├── hy_armtdp-ud-dev.conllu
│   └── hy_armtdp-ud-test.conllu
├── src/                       # Core source code
│   ├── hmm.py                # Main HMM implementation
│   ├── hmm_trigram.py        # Trigram (2nd-order) HMM
│   ├── data_utils.py         # Data loading utilities
│   ├── evaluation.py         # Evaluation and analysis
│   ├── config_loader.py      # Configuration loader
│   └── main.py               # Main training script
├── examples/                  # Example scripts
│   ├── basic_usage.py        # Simple usage example
│   ├── compare_models.py     # Compare HMM variants
│   └── tag_custom_text.py    # Tag custom Armenian text
├── visualization/             # Visualization module
│   ├── visualization.py      # Visualization library
│   ├── visualize_all.py      # Generate all visualizations
│   ├── VISUALIZATION.md      # Original documentation
│   └── VISUALIZATION_GUIDE.md # Comprehensive guide
├── visualizations/            # Generated plots (created at runtime)
├── models/                    # Saved trained models
├── config.cfg                 # Configuration file
├── requirements.txt           # Python dependencies
├── DATASET.md                # Dataset documentation
└── README.md                 # This file
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

This creates comprehensive visualizations including:
- Model comparison charts
- Suffix pattern analysis
- Known vs unknown word performance
- State transition graphs and matrices
- Confusion matrices
- Viterbi decoding examples

See [VISUALIZATION.md](visualization/VISUALIZATION.md) for detailed documentation.

## Features

✅ **Complete HMM Implementation**
- Object-oriented design with clean API
- Maximum Likelihood Estimation for parameters
- Viterbi algorithm for decoding
- Add-epsilon smoothing for unseen events

✅ **Data Processing**
- CoNLL-U file parsing
- Armenian Universal Dependencies dataset support
- Automatic train/dev/test split handling

✅ **Evaluation & Analysis**
- Accuracy computation
- Error analysis with examples
- Confusion matrix generation
- Per-tag statistics

✅ **Visualization**
- State transition graphs
- Transition/emission matrix heatmaps
- Viterbi path visualization
- Initial probability distributions

✅ **Model Persistence**
- Save/load trained models
- Pickle-based serialization

## HMM Components

The model learns four key components:

1. **Q** - Set of POS tags (states)
   - Example: {NOUN, VERB, ADJ, ADP, PUNCT, ...}

2. **π** (Pi) - Initial state probabilities
   - P(sentence starts with tag)
   
3. **A** - Transition probability matrix
   - A[i][j] = P(tag_j | tag_i)
   - Probability of transitioning from one tag to another

4. **B** - Emission probability matrix
   - B[i][j] = P(word_j | tag_i)
   - Probability of observing a word given a tag

## Performance

Performance on Armenian UD dataset is evaluated using accuracy metric.

## File Descriptions

### Core Files
- **`config.cfg`** - Configuration file for dataset paths and output settings
- **`src/hmm.py`** - HiddenMarkovModel class with train/predict/evaluate methods
- **`src/data_utils.py`** - Functions for loading and processing CoNLL-U datasets
- **`src/evaluation.py`** - Evaluation metrics, error analysis, and reporting
- **`src/config_loader.py`** - Utility to load and parse configuration file
- **`src/main.py`** - Main training and evaluation pipeline

### Visualization (Optional)
- **`visualization/visualization.py`** - All visualization functions for HMM analysis
- **`visualization/visualize_hmm.py`** - Generate all visualizations
- **`visualization/VISUALIZATION.md`** - Documentation for visualization features

### Examples
- **`examples/basic_usage.py`** - Usage examples and demonstrations

## References

- Armenian UD Treebank: https://github.com/UniversalDependencies/UD_Armenian-ArmTDP

## License

Educational project for ASDS25 Statistics course.
