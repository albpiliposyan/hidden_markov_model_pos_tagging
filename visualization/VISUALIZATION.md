# HMM Visualization Guide

## Installation

First, install the required visualization libraries:

```bash
pip install matplotlib seaborn networkx
```

## Quick Start

### Option 1: Run the Full Visualization Demo

```bash
python src/visualize_hmm.py
```

This will:
1. Load and train the HMM
2. Create all visualizations interactively
3. Save them to the `visualizations/` folder

### Option 2: Use Individual Visualizations

```python
from hmm import HiddenMarkovModel
from data_utils import load_armenian_dataset
from visualization import *

# Load and train
train_data, dev_data, test_data = load_armenian_dataset()
hmm = HiddenMarkovModel()
hmm.train(train_data)

# Create specific visualizations
visualize_transition_graph(hmm, min_prob=0.05)
visualize_transition_matrix(hmm, top_n=12)
visualize_initial_probabilities(hmm)
visualize_emission_probabilities(hmm, tag='NOUN', top_n=20)
visualize_confusion_matrix(hmm, test_data, top_n=10)

# Visualize Viterbi decoding for a sentence
words = ["Մտածում", "եմ", "՝", "Ադամի", "ու", "Եվայի"]
true_tags = ["VERB", "AUX", "PUNCT", "PROPN", "CCONJ", "PROPN"]
visualize_viterbi_path(hmm, words, true_tags)
```

### Option 3: Generate All Visualizations and Save

```python
from hmm import HiddenMarkovModel
from data_utils import load_armenian_dataset
from visualization import create_all_visualizations

# Load and train
train_data, dev_data, test_data = load_armenian_dataset()
hmm = HiddenMarkovModel()
hmm.train(train_data)

# Create all visualizations
create_all_visualizations(hmm, test_data, output_dir='my_visualizations')
```

## Available Visualizations

### 1. **Transition Graph (State Machine)**
```python
visualize_transition_graph(hmm, min_prob=0.05)
```
Shows the HMM as a directed graph where:
- **Nodes** = POS tags (states)
- **Node size** = Initial probability (π)
- **Edges** = Transitions with probability > threshold
- **Edge thickness** = Transition probability strength
- **Edge labels** = Exact transition probabilities

### 2. **Transition Matrix Heatmap**
```python
visualize_transition_matrix(hmm, top_n=15)
```
Shows transition probabilities A as a heatmap:
- **Rows** = Current state
- **Columns** = Next state
- **Color intensity** = Probability strength
- Shows the top N most common tags

### 3. **Initial Probabilities**
```python
visualize_initial_probabilities(hmm)
```
Bar chart showing P(sentence starts with each tag):
- Highlights which tags commonly start sentences
- Top 3 tags are highlighted in a different color

### 4. **Emission Probabilities**
```python
visualize_emission_probabilities(hmm, tag='NOUN', top_n=20)
```
Shows the top N words most likely to be emitted by a specific tag:
- Horizontal bar chart
- Helps understand what words characterize each POS tag

### 5. **Confusion Matrix**
```python
visualize_confusion_matrix(hmm, test_data, top_n=12)
```
Shows how often tags are confused with each other:
- **Diagonal** = Correct predictions
- **Off-diagonal** = Confusion between tags
- Normalized to show proportions

### 6. **Viterbi Path Visualization**
```python
visualize_viterbi_path(hmm, words, true_tags)
```
Shows the step-by-step tagging process:
- **Blue boxes** = Words
- **Green boxes** = Correct predictions
- **Red boxes** = Incorrect predictions
- **Arrows** = State transitions

## Visualization Parameters

### Common Parameters

- **`figsize`**: Tuple (width, height) in inches
  ```python
  visualize_transition_graph(hmm, figsize=(20, 15))
  ```

- **`save_path`**: Path to save the figure
  ```python
  visualize_transition_matrix(hmm, save_path='my_plot.png')
  ```

- **`top_n`**: Number of items to display
  ```python
  visualize_transition_matrix(hmm, top_n=20)
  ```

### Specific Parameters

- **`min_prob`** (transition graph): Minimum probability to show edges
  ```python
  visualize_transition_graph(hmm, min_prob=0.1)  # Only strong transitions
  ```

- **`tag`** (emission probabilities): Which POS tag to analyze
  ```python
  visualize_emission_probabilities(hmm, tag='VERB', top_n=25)
  ```

## Example Output

After running `create_all_visualizations()`, you'll get:

```
visualizations/
├── transition_graph.png          # State machine diagram
├── transition_matrix.png         # Heatmap of transitions
├── initial_probabilities.png     # Starting state distribution
├── emissions_NOUN.png            # Top NOUN words
├── emissions_VERB.png            # Top VERB words
├── emissions_ADJ.png             # Top ADJ words
├── emissions_ADP.png             # Top ADP words
├── confusion_matrix.png          # Error analysis
└── viterbi_path_sample.png       # Example tagging process
```

## Tips

1. **For presentations**: Use high DPI
   ```python
   plt.savefig('plot.png', dpi=300, bbox_inches='tight')
   ```

2. **For large graphs**: Increase min_prob threshold
   ```python
   visualize_transition_graph(hmm, min_prob=0.1)  # Cleaner graph
   ```

3. **Interactive mode**: Don't use save_path to see plots interactively
   ```python
   visualize_transition_graph(hmm)  # Opens in window
   ```

4. **Batch mode**: Save all without showing
   ```python
   import matplotlib
   matplotlib.use('Agg')  # No display
   create_all_visualizations(hmm, test_data)
   ```

## Understanding the Visualizations

### What to Look For

**Transition Graph:**
- Thick edges = Common transitions
- Loops = Tags that often repeat
- Central nodes = High connectivity tags

**Transition Matrix:**
- Bright diagonal = Self-transitions (tag → same tag)
- Bright off-diagonal = Common sequences
- Dark areas = Unlikely transitions

**Confusion Matrix:**
- Perfect model = All values on diagonal
- Similar tags = Confused (e.g., NOUN vs PROPN)
- Asymmetric confusion = One-way errors

**Emission Probabilities:**
- High bars = Characteristic words for that tag
- Helps verify if model learned correctly
