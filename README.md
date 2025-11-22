# Hidden Markov Model - POS Tagger for Armenian

A complete implementation of Hidden Markov Models for Part-of-Speech tagging using the Armenian Universal Dependencies dataset.

## Project Structure

```
hidden_markov_model/
├── dataset/                   # CoNLL-U format datasets
│   ├── hy_armtdp-ud-train.conllu
│   ├── hy_armtdp-ud-dev.conllu
│   └── hy_armtdp-ud-test.conllu
├── src/                       # Source code
│   ├── hmm.py                # HMM class implementation
│   ├── data_utils.py         # Data loading utilities
│   ├── evaluation.py         # Evaluation and analysis tools
│   ├── config_loader.py      # Configuration file loader
│   └── main.py               # Main training script
├── visualization/             # Visualization tools (optional)
│   ├── visualization.py      # Visualization functions
│   ├── visualize_hmm.py      # Visualization demo script
│   └── VISUALIZATION.md      # Visualization documentation
├── examples/                  # Usage examples
│   └── basic_usage.py
├── models/                    # Saved trained models
├── config.cfg                 # Configuration file
├── requirements.txt           # Python dependencies
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
# Using default config.cfg
python src/main.py

# Or specify a custom config file
python src/main.py my_config.cfg
```

This will:
- Load the Armenian UD dataset
- Train the HMM model with parameters from config.cfg
- Evaluate on dev and test sets
- Display sample predictions
- Perform error analysis
- Save the trained model

### 3. Configuration

Edit `config.cfg` to customize training and evaluation:

```ini
[dataset]
train_path = dataset/hy_armtdp-ud-train.conllu
dev_path = dataset/hy_armtdp-ud-dev.conllu
test_path = dataset/hy_armtdp-ud-test.conllu

[training]
random_seed = 42

[output]
save_model = True
model_path = models/hmm_armenian_pos.pkl
```

See `config.cfg` for all available options.

### 4. Visualize the HMM

```bash
```bash
python visualization/visualize_hmm.py
```

This creates visualizations including:
- State transition graph
- Transition matrix heatmap
- Emission probabilities
- Confusion matrix
- Viterbi path examples

See [VISUALIZATION.md](visualization/VISUALIZATION.md) for detailed documentation.

## Usage Examples

### Training a Model

```python
from src.hmm import HiddenMarkovModel
from src.data_utils import load_armenian_dataset

# Load data
train_data, dev_data, test_data = load_armenian_dataset()

# Train HMM
hmm = HiddenMarkovModel()
hmm.train(train_data)

# Save model
hmm.save('models/my_hmm.pkl')
```

### Making Predictions

```python
# Predict POS tags for a sentence
words = ["Մտածում", "եմ", "՝", "Ադամի", "ու", "Եվայի"]
tags = hmm.predict(words)
print(tags)
# Output: ['VERB', 'AUX', 'PUNCT', 'PROPN', 'CCONJ', 'PROPN']
```

### Evaluating the Model

```python
from src.evaluation import evaluate_and_report

# Evaluate on test set
results = evaluate_and_report(hmm, test_data, "Test")
print(f"Accuracy: {results['accuracy']:.4f}")
```

### Loading a Saved Model

```python
# Load pre-trained model
hmm = HiddenMarkovModel.load('models/hmm_armenian_pos.pkl')

# Use it directly
words = ["some", "words"]
tags = hmm.predict(words)
```

### Creating Visualizations

```python
from visualization.visualization import visualize_transition_graph, create_all_visualizations

# Create transition graph
visualize_transition_graph(hmm, min_prob=0.05)

# Or create all visualizations at once
create_all_visualizations(hmm, test_data, output_dir='visualizations')
```

See `visualization/VISUALIZATION.md` for more details.

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

Expected accuracy on Armenian UD dataset: **~85-92%**

Actual performance depends on:
- Smoothing parameter
- Vocabulary coverage
- Dataset size
- Tag complexity

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

## Future Improvements

### Potential Enhancements

- [ ] Log-space computation to prevent numerical underflow
- [ ] Forward-Backward algorithm for marginal probabilities
- [ ] Baum-Welch (EM) algorithm for unsupervised training
- [ ] Better OOV word handling (morphological features, character n-grams)
- [ ] Per-tag precision, recall, F1 scores
- [ ] Cross-validation support
- [ ] Hyperparameter tuning (grid search)
- [ ] Command-line interface with argparse
- [ ] Unit tests and CI/CD
- [ ] Jupyter notebook tutorial

## References

- Universal Dependencies: https://universaldependencies.org/
- Armenian UD Treebank: https://github.com/UniversalDependencies/UD_Armenian-ArmTDP
- Speech and Language Processing (Jurafsky & Martin) - Chapter 8: POS Tagging

## Contributing

This is an educational project. Feel free to:
- Report issues
- Suggest improvements
- Add features
- Improve documentation

## License

Educational project for Statistical NLP coursework.
