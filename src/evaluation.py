"""Evaluation utilities for HMM POS tagging."""

import numpy as np


def evaluate_and_report(hmm, test_data, dataset_name="Test", method="viterbi"):
    """Evaluate HMM and print detailed report."""
    results = hmm.evaluate(test_data, method=method)
    
    print(f"\n{dataset_name} Set Results:")
    print(f"  Total tokens: {results['total_tokens']}")
    print(f"  Correct predictions: {results['correct']}")
    print(f"  Incorrect predictions: {results['incorrect']}")
    print(f"  Accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
    
    return results


def compute_confusion_matrix(hmm, test_data):
    """Compute confusion matrix for POS tagging."""
    n_tags = len(hmm.states)
    confusion_matrix = np.zeros((n_tags, n_tags), dtype=int)
    
    for sentence in test_data:
        words = [word for word, pos in sentence]
        true_tags = [pos for word, pos in sentence]
        
        predicted_tags = hmm.predict(words)
        
        for true_tag, pred_tag in zip(true_tags, predicted_tags):
            true_idx = hmm.tag_to_idx[true_tag]
            pred_idx = hmm.tag_to_idx[pred_tag]
            confusion_matrix[true_idx][pred_idx] += 1
    
    return confusion_matrix


def print_confusion_matrix(confusion_matrix, tags, top_n=10):
    """Print confusion matrix for top N most common tags."""
    # Get top tags by frequency
    tag_totals = confusion_matrix.sum(axis=1)
    top_indices = np.argsort(tag_totals)[::-1][:top_n]
    
    print(f"\nConfusion Matrix (Top {top_n} tags):")
    print(f"{'':10s}", end='')
    for idx in top_indices:
        print(f"{tags[idx]:10s}", end='')
    print()
    
    for i in top_indices:
        print(f"{tags[i]:10s}", end='')
        for j in top_indices:
            print(f"{confusion_matrix[i][j]:10d}", end='')
        print()


def analyze_errors(hmm, test_data, max_errors=10):
    """Analyze and print common errors."""
    error_examples = []
    
    for sentence in test_data:
        words = [word for word, pos in sentence]
        true_tags = [pos for word, pos in sentence]
        
        predicted_tags = hmm.predict(words)
        
        for i, (word, true_tag, pred_tag) in enumerate(zip(words, true_tags, predicted_tags)):
            if true_tag != pred_tag:
                context_start = max(0, i - 2)
                context_end = min(len(words), i + 3)
                context = ' '.join(words[context_start:context_end])
                
                error_examples.append({
                    'word': word,
                    'true_tag': true_tag,
                    'pred_tag': pred_tag,
                    'context': context
                })
                
                if len(error_examples) >= max_errors:
                    break
        
        if len(error_examples) >= max_errors:
            break
    
    print(f"\n{'='*70}")
    print(f"ERROR ANALYSIS (First {len(error_examples)} errors)")
    print(f"{'='*70}")
    
    for i, error in enumerate(error_examples, 1):
        print(f"\n{i}. Word: '{error['word']}'")
        print(f"   True tag: {error['true_tag']} | Predicted: {error['pred_tag']}")
        print(f"   Context: ...{error['context']}...")


def print_sample_predictions(hmm, test_data, n_samples=1, max_tokens=15):
    """Print sample predictions from test data."""
    for sample_idx in range(min(n_samples, len(test_data))):
        print(f"\n{'='*70}")
        print(f"SAMPLE PREDICTION #{sample_idx + 1}")
        print(f"{'='*70}")
        
        sample_sentence = test_data[sample_idx]
        words = [word for word, pos in sample_sentence]
        true_tags = [pos for word, pos in sample_sentence]
        
        predicted_tags = hmm.predict(words)
        
        print(f"\nSentence with {len(words)} words:")
        print(f"\n{'№':3s} {'Word':20s} {'True Tag':10s} {'Predicted':10s} {'Match':5s}")
        print("-" * 70)
        
        for i, (word, true_tag, pred_tag) in enumerate(zip(words, true_tags, predicted_tags), 1):
            match = "✓" if true_tag == pred_tag else "✗"
            print(f"{i:3d} {word[:20]:20s} {true_tag:10s} {pred_tag:10s} {match:5s}")
            
            if i >= max_tokens:
                if len(words) > max_tokens:
                    print(f"... and {len(words) - max_tokens} more tokens")
                break


def print_hmm_summary(hmm, verbose=False):
    """Print summary of trained HMM."""
    print(f"\n{'='*70}")
    print("HMM SUMMARY")
    print(f"{'='*70}")
    
    info = hmm.get_info()
    print(f"\nModel Configuration:")
    print(f"  Number of states (POS tags): {info['n_states']}")
    print(f"  Vocabulary size: {info['n_words']}")
    print(f"  Transition matrix shape: {info['transition_matrix_shape']}")
    print(f"  Emission matrix shape: {info['emission_matrix_shape']}")
    
    print(f"\nPOS Tags: {info['states']}")
    
    if verbose:
        # Top initial probabilities
        print(f"\nTop 5 Initial Probabilities:")
        top_initial = sorted(enumerate(hmm.initial_probs), key=lambda x: x[1], reverse=True)[:5]
        for idx, prob in top_initial:
            print(f"  {hmm.idx_to_tag[idx]:10s}: {prob:.4f}")
    
        # Sample transitions
        if 'NOUN' in hmm.tag_to_idx and 'VERB' in hmm.tag_to_idx:
            noun_idx = hmm.tag_to_idx['NOUN']
            verb_idx = hmm.tag_to_idx['VERB']
            print(f"\nSample Transition Probabilities:")
            print(f"  P(VERB | NOUN) = {hmm.transition_probs[noun_idx][verb_idx]:.4f}")
            print(f"  P(NOUN | VERB) = {hmm.transition_probs[verb_idx][noun_idx]:.4f}")
    
        # Top emissions for a sample tag
        if 'NOUN' in hmm.tag_to_idx:
            noun_idx = hmm.tag_to_idx['NOUN']
            top_emissions = sorted(enumerate(hmm.emission_probs[noun_idx]), key=lambda x: x[1], reverse=True)[:5]
            print(f"\nTop 5 words emitted by NOUN:")
            for word_idx, prob in top_emissions:
                print(f"  '{hmm.idx_to_word[word_idx]}': {prob:.4f}")
    
    print(f"\n{'='*70}")


def compare_models(results_dict):
    """Compare multiple model results."""
    print(f"\n{'='*70}")
    print("MODEL COMPARISON")
    print(f"{'='*70}")
    print(f"\n{'Model':20s} {'Accuracy':>10s} {'Correct':>10s} {'Total':>10s}")
    print("-" * 70)
    
    for model_name, results in results_dict.items():
        acc = results['accuracy']
        correct = results['correct']
        total = results['total_tokens']
        print(f"{model_name:20s} {acc:>10.4f} {correct:>10d} {total:>10d}")
