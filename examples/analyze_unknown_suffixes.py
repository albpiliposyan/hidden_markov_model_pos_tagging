"""Analyze unknown suffix statistics for different suffix lengths."""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from hmm import HiddenMarkovModel
from data_utils import load_armenian_dataset


def main():
    print("=" * 70)
    print("UNKNOWN SUFFIX ANALYSIS")
    print("=" * 70)
    
    # Load data
    print("\nLoading data...")
    train_data, dev_data, test_data = load_armenian_dataset()
    
    combined_train_data = train_data + dev_data
    
    print(f"Training sentences: {len(combined_train_data)}")
    print(f"Test sentences: {len(test_data)}")
    
    # Test different suffix lengths
    suffix_lengths = [2, 3, 4]
    
    for n in suffix_lengths:
        print("\n" + "=" * 70)
        print(f"SUFFIX LENGTH n={n}")
        print("=" * 70)
        
        # Train model
        print(f"\nTraining Suffix-Enhanced HMM (n={n})...")
        hmm = HiddenMarkovModel(use_suffix_model=True, suffix_length=n)
        hmm.train(combined_train_data)
        
        # Get suffix statistics
        print(f"\nAnalyzing unknown words in test set...")
        stats = hmm.unknown_suffix_statistics(test_data)
        
        if stats:
            print(f"\nUnknown Word Suffix Statistics:")
            print(f"  Total unknown words: {stats['total_unknown_words']}")
            print(f"  Words with KNOWN suffixes: {stats['known_suffix_count']} ({stats['known_suffix_percent']:.2f}%)")
            print(f"  Words with UNKNOWN suffixes: {stats['unknown_suffix_count']} ({stats['unknown_suffix_percent']:.2f}%)")
            print(f"  Words too short for suffix: {stats['too_short_count']} ({stats['too_short_percent']:.2f}%)")
            
            # Evaluate accuracy
            results = hmm.evaluate(test_data)
            print(f"\nOverall Accuracy: {results['accuracy']*100:.2f}%")
    
    print("\n" + "=" * 70)


if __name__ == '__main__':
    main()
