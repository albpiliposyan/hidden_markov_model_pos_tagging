"""
Simple examples demonstrating HMM usage.
Run this to see basic functionality.
"""

import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from hmm import HiddenMarkovModel
from data_utils import load_armenian_dataset


def example_1_train_and_predict():
    """Example 1: Train a model and make predictions."""
    print("\n" + "="*70)
    print("EXAMPLE 1: Train HMM and Make Predictions")
    print("="*70)
    
    # Load data
    print("\nLoading data...")
    train_data, dev_data, test_data = load_armenian_dataset()
    
    # Train model
    print("\nTraining HMM...")
    hmm = HiddenMarkovModel()
    hmm.train(train_data)
    
    # Make predictions
    print("\nMaking predictions...")
    sample_sentence = test_data[0]
    words = [w for w, t in sample_sentence][:10]
    true_tags = [t for w, t in sample_sentence][:10]
    
    predicted_tags = hmm.predict(words)
    
    print(f"\nSample sentence ({len(words)} words):")
    print(f"\n{'Word':<20} {'True Tag':<10} {'Predicted':<10} {'Match'}")
    print("-" * 55)
    
    for word, true, pred in zip(words, true_tags, predicted_tags):
        match = "✓" if true == pred else "✗"
        print(f"{word[:20]:<20} {true:<10} {pred:<10} {match}")


def example_2_evaluate():
    """Example 2: Evaluate model performance."""
    print("\n" + "="*70)
    print("EXAMPLE 2: Evaluate Model Performance")
    print("="*70)
    
    # Load data
    train_data, dev_data, test_data = load_armenian_dataset()
    
    # Train model
    print("\nTraining HMM...")
    hmm = HiddenMarkovModel()
    hmm.train(train_data)
    
    # Evaluate
    print("\nEvaluating on test set...")
    results = hmm.evaluate(test_data)
    
    print(f"\nTest Results:")
    print(f"  Total tokens: {results['total_tokens']}")
    print(f"  Correct: {results['correct']}")
    print(f"  Incorrect: {results['incorrect']}")
    print(f"  Accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")


def example_3_save_load():
    """Example 3: Save and load a trained model."""
    print("\n" + "="*70)
    print("EXAMPLE 3: Save and Load Model")
    print("="*70)
    
    # Train a simple model
    train_data, _, test_data = load_armenian_dataset()
    
    print("\nTraining HMM...")
    hmm = HiddenMarkovModel()
    hmm.train(train_data)
    
    # Save model
    model_path = 'models/example_hmm.pkl'
    print(f"\nSaving model to {model_path}...")
    hmm.save(model_path)
    
    # Load model
    print(f"\nLoading model from {model_path}...")
    loaded_hmm = HiddenMarkovModel.load(model_path)
    
    # Test loaded model
    print("\nTesting loaded model...")
    sample = test_data[0]
    words = [w for w, t in sample][:5]
    tags = loaded_hmm.predict(words)
    
    print(f"\nPrediction from loaded model:")
    for word, tag in zip(words, tags):
        print(f"  '{word}' -> {tag}")


def example_4_model_info():
    """Example 4: Inspect model parameters."""
    print("\n" + "="*70)
    print("EXAMPLE 4: Inspect Model Parameters")
    print("="*70)
    
    # Train model
    train_data, _, _ = load_armenian_dataset()
    
    print("\nTraining HMM...")
    hmm = HiddenMarkovModel()
    hmm.train(train_data)
    
    # Get model info
    info = hmm.get_info()
    
    print(f"\nModel Information:")
    print(f"  Number of states: {info['n_states']}")
    print(f"  Vocabulary size: {info['n_words']}")
    print(f"  Smoothing: {info['smoothing']}")
    
    print(f"\nPOS Tags ({info['n_states']} total):")
    print(f"  {info['states']}")
    
    print(f"\nTop 5 Initial Probabilities:")
    import numpy as np
    top_indices = np.argsort(hmm.pi)[::-1][:5]
    for idx in top_indices:
        tag = hmm.idx_to_tag[idx]
        prob = hmm.pi[idx]
        print(f"  {tag:10s}: {prob:.4f}")


def main():
    """Run all examples."""
    print("="*70)
    print("HMM POS Tagger - Usage Examples")
    print("="*70)
    
    examples = [
        example_1_train_and_predict,
        example_2_evaluate,
        example_3_save_load,
        example_4_model_info,
    ]
    
    for i, example_func in enumerate(examples, 1):
        try:
            example_func()
        except Exception as e:
            print(f"\nExample {i} failed with error: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*70)
    print("All examples completed!")
    print("="*70)


if __name__ == "__main__":
    main()
