"""Generate all visualizations for HMM project."""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from hmm import HiddenMarkovModel
from hmm_trigram import TrigramHMM
from data_utils import load_armenian_dataset
from visualization import (
    visualize_model_comparison,
    visualize_suffix_patterns,
    visualize_unknown_word_performance,
    compare_models_on_word_types,
    create_all_visualizations
)


def main():
    """Generate all visualizations."""
    
    print("="*70)
    print("GENERATING ALL HMM VISUALIZATIONS")
    print("="*70)
    
    # Load Data
    print("\n[1] Loading Dataset...")
    train_data, dev_data, test_data = load_armenian_dataset()
    combined_train_data = train_data + dev_data
    
    print(f"  Training: {len(combined_train_data)} sentences")
    print(f"  Test: {len(test_data)} sentences")
    
    # Train all models
    print("\n[2] Training Models...")
    
    print("  Training Basic HMM...")
    basic_hmm = HiddenMarkovModel(use_suffix_model=False)
    basic_hmm.train(combined_train_data)
    basic_results = basic_hmm.evaluate(test_data)
    
    print("  Training Suffix-Enhanced HMM (n=2)...")
    suffix_hmm_n2 = HiddenMarkovModel(use_suffix_model=True, suffix_length=2)
    suffix_hmm_n2.train(combined_train_data)
    suffix_results_n2 = suffix_hmm_n2.evaluate(test_data)
    
    print("  Training Suffix-Enhanced HMM (n=3)...")
    suffix_hmm_n3 = HiddenMarkovModel(use_suffix_model=True, suffix_length=3)
    suffix_hmm_n3.train(combined_train_data)
    suffix_results_n3 = suffix_hmm_n3.evaluate(test_data)
    
    print("  Training Suffix-Enhanced HMM (n=4)...")
    suffix_hmm_n4 = HiddenMarkovModel(use_suffix_model=True, suffix_length=4)
    suffix_hmm_n4.train(combined_train_data)
    suffix_results_n4 = suffix_hmm_n4.evaluate(test_data)
    
    # Note: Trigram model evaluation is very slow, using pre-computed accuracy
    print("  Using pre-computed Trigram HMM accuracy (68.24%)...")
    trigram_accuracy = 0.6824
    
    # Use the best suffix model for detailed visualizations
    suffix_hmm = suffix_hmm_n3
    
    # Create output directory
    output_dir = 'visualizations'
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate new visualizations
    print("\n[3] Generating New Visualizations...")
    
    # Model comparison with different suffix lengths
    print("\n  Creating model comparison chart...")
    models_dict = {
        'Classical HMM': basic_results['accuracy'],
        'Suffix n=2': suffix_results_n2['accuracy'],
        'Suffix n=3': suffix_results_n3['accuracy'],
        'Suffix n=4': suffix_results_n4['accuracy'],
        'Trigram HMM': trigram_accuracy
    }
    visualize_model_comparison(models_dict, 
                              save_path=f'{output_dir}/model_comparison.png')
    
    # Suffix patterns (for suffix-enhanced model)
    print("  Creating suffix pattern analysis...")
    visualize_suffix_patterns(suffix_hmm, top_n=30,
                             save_path=f'{output_dir}/suffix_patterns.png')
    
    # Unknown word performance (for suffix-enhanced model)
    print("  Creating unknown word performance analysis...")
    stats = visualize_unknown_word_performance(suffix_hmm, test_data,
                                              save_path=f'{output_dir}/unknown_word_performance.png')
    
    print(f"\n  Known words accuracy: {stats['known_accuracy']:.2f}%")
    print(f"  Unknown words accuracy: {stats['unknown_accuracy']:.2f}%")
    
    # Compare models on known vs unknown words
    print("  Creating model comparison on word types...")
    models_for_comparison = {
        'Classical HMM': basic_hmm,
        'Suffix n=2': suffix_hmm_n2,
        'Suffix n=3': suffix_hmm_n3,
        'Suffix n=4': suffix_hmm_n4
    }
    compare_results = compare_models_on_word_types(models_for_comparison, test_data,
                                                   save_path=f'{output_dir}/models_word_types_comparison.png')
    
    print("\n  Comparison Results:")
    for model_name, accs in compare_results.items():
        print(f"    {model_name}:")
        print(f"      Known: {accs['known']:.2f}%")
        print(f"      Unknown: {accs['unknown']:.2f}%")
    
    # Generate all standard visualizations for best model (suffix-enhanced)
    print("\n[4] Generating All Standard Visualizations (Suffix-Enhanced Model)...")
    create_all_visualizations(suffix_hmm, test_data, output_dir)
    
    # Summary
    print("\n" + "="*70)
    print("VISUALIZATION SUMMARY")
    print("="*70)
    print(f"\nAll visualizations saved to '{output_dir}/' directory:")
    print("  1. model_comparison.png - Compare all three models")
    print("  2. suffix_patterns.png - Top 30 suffix patterns learned")
    print("  3. unknown_word_performance.png - Known vs unknown word accuracy")
    print("  4. models_word_types_comparison.png - Basic vs Suffix-Enhanced on word types")
    print("  5. transition_graph.png - State transition network")
    print("  6. transition_matrix.png - Transition probability heatmap")
    print("  7. initial_probabilities.png - Initial state probabilities")
    print("  8. confusion_matrix.png - Prediction confusion matrix")
    print("  9. viterbi_path_sample.png - Sample Viterbi decoding path")
    print("\n" + "="*70)


if __name__ == "__main__":
    main()
