"""Compare HMM with and without suffix-based emission probabilities."""

import os
import sys
sys.path.insert(0, 'src')

from hmm import HiddenMarkovModel
from hmm_trigram import TrigramHMM
from data_utils import load_armenian_dataset


def count_unknown_words(model, test_data):
    """Count unknown words in test data for a given model."""
    unknown_count = 0
    total_count = 0
    
    for sentence in test_data:
        for word, _ in sentence:
            total_count += 1
            if word not in model.word_to_idx:
                unknown_count += 1
    
    return unknown_count, total_count


def main():
    """Compare basic HMM vs suffix-enhanced HMM vs trigram HMM."""
    
    print("\n" + "="*70)
    print("HMM COMPARISON: Basic vs Suffix-Enhanced vs Trigram")
    print("="*70)
    
    # Load Data
    print("\n[1] Loading Dataset...")
    train_data, dev_data, test_data = load_armenian_dataset()
    combined_train_data = train_data + dev_data
    
    print(f"  Training: {len(combined_train_data)} sentences")
    print(f"  Test: {len(test_data)} sentences")
    
    # Train Basic HMM (without suffix model)
    print("\n[2] Training Basic HMM (uniform unknown word handling)...")
    basic_hmm = HiddenMarkovModel(use_suffix_model=False)
    basic_hmm.train(combined_train_data)
    
    # Train Suffix-Enhanced HMM
    print("\n[3] Training Suffix-Enhanced HMM (bigram)...")
    suffix_hmm = HiddenMarkovModel(use_suffix_model=True)
    suffix_hmm.train(combined_train_data)
    
    # Train Trigram HMM
    print("\n[4] Training Trigram HMM (second-order)...")
    trigram_hmm = TrigramHMM()
    trigram_hmm.train(combined_train_data)
    
    # Evaluate All Models
    print("\n[5] Evaluating All Models on Test Set...")
    print("\n" + "-"*70)
    
    # Count unknown words (same for all models since same vocabulary)
    unknown_count, total_count = count_unknown_words(basic_hmm, test_data)
    unknown_rate = (unknown_count / total_count * 100) if total_count > 0 else 0
    
    print(f"\nTest Set Info:")
    print(f"  Total tokens: {total_count}")
    print(f"  Unknown words: {unknown_count} ({unknown_rate:.2f}%)")
    print()
    
    # Basic HMM
    basic_results = basic_hmm.evaluate(test_data)
    print("Basic HMM (Bigram) Results:")
    print(f"  Accuracy: {basic_results['accuracy']:.4f} ({basic_results['accuracy']*100:.2f}%)")
    print(f"  Correct: {basic_results['correct']}/{basic_results['total_tokens']}")
    
    # Suffix-Enhanced HMM
    suffix_results = suffix_hmm.evaluate(test_data)
    print("\nSuffix-Enhanced HMM (Bigram) Results:")
    print(f"  Accuracy: {suffix_results['accuracy']:.4f} ({suffix_results['accuracy']*100:.2f}%)")
    print(f"  Correct: {suffix_results['correct']}/{suffix_results['total_tokens']}")
    
    # Trigram HMM
    trigram_results = trigram_hmm.evaluate(test_data)
    print("\nTrigram HMM (Second-Order) Results:")
    print(f"  Accuracy: {trigram_results['accuracy']:.4f} ({trigram_results['accuracy']*100:.2f}%)")
    print(f"  Correct: {trigram_results['correct']}/{trigram_results['total_tokens']}")
    
    # Comparison
    print("\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70)
    print(f"\nBasic HMM (Bigram):           {basic_results['accuracy']*100:.2f}%")
    print(f"Suffix-Enhanced HMM (Bigram): {suffix_results['accuracy']*100:.2f}%")
    print(f"Trigram HMM (Second-Order):   {trigram_results['accuracy']*100:.2f}%")
    
    best_accuracy = max(basic_results['accuracy'], suffix_results['accuracy'], trigram_results['accuracy'])
    if best_accuracy == suffix_results['accuracy']:
        winner = "Suffix-Enhanced HMM (Bigram)"
    elif best_accuracy == trigram_results['accuracy']:
        winner = "Trigram HMM"
    else:
        winner = "Basic HMM (Bigram)"
    
    print(f"\nBest Model: {winner} with {best_accuracy*100:.2f}% accuracy")
    
    improvement_suffix = (suffix_results['accuracy'] - basic_results['accuracy']) * 100
    print(f"\nSuffix enhancement improvement: +{improvement_suffix:.2f} percentage points")
    print(f"  ({basic_results['incorrect'] - suffix_results['incorrect']} fewer errors)")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    main()
