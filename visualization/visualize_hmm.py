"""Demo script for HMM visualizations."""

import sys
import os

# Add parent directories to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.dirname(__file__))

from hmm import HiddenMarkovModel
from data_utils import load_armenian_dataset
from visualization import (
    visualize_transition_graph,
    visualize_transition_matrix,
    visualize_emission_probabilities,
    visualize_initial_probabilities,
    visualize_viterbi_path,
    visualize_confusion_matrix,
    create_all_visualizations
)


def main():
    print("="*70)
    print("HMM Visualization Demo")
    print("="*70)
    
    # Load data
    print("\n[1] Loading data...")
    train_data, dev_data, test_data = load_armenian_dataset()
    
    # Train HMM (or load existing model)
    print("\n[2] Training HMM...")
    hmm = HiddenMarkovModel()
    hmm.train(train_data)
    
    # Create all visualizations and save them
    print("\n[3] Generating all visualizations and saving to disk...")
    create_all_visualizations(hmm, test_data, output_dir='visualizations')
    
    print("\n" + "="*70)
    print("Visualization demo complete!")
    print("Check the 'visualizations/' folder for saved images.")
    print("="*70)


if __name__ == "__main__":
    main()
