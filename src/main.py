"""Main script for training and evaluating HMM-based POS tagger."""

import os
from hmm import HiddenMarkovModel
from data_utils import load_armenian_dataset, print_dataset_stats
from evaluation import (
    evaluate_and_report, 
    print_sample_predictions, 
    print_hmm_summary
)
from config_loader import (
    load_config,
    get_training_config,
    get_output_config
)


def main(config_path='config.cfg'):
    """Main execution function."""
    
    # Load configuration
    if os.path.exists(config_path):
        config = load_config(config_path)
        training_cfg = get_training_config(config)
        output_cfg = get_output_config(config)
    else:
        print(f"Warning: Config file '{config_path}' not found. Using defaults.")
        training_cfg = {'save_model': True, 'model_name': 'hmm_armenian_pos.pkl', 'model_dir': 'models'}
        output_cfg = {'verbose': True, 'print_hmm_summary': True, 'print_dataset_stats': True}
    
    print("\n" + "="*70)
    print("HMM-based POS Tagger for Armenian language")
    print("="*70)
    
    # Load Data
    print("\n[1] Loading Dataset...")
    train_data, dev_data, test_data = load_armenian_dataset()
    
    # Combine train and dev datasets
    combined_train_data = train_data + dev_data
    print(f"\nCombining training and development sets:")
    print(f"  Original training: {len(train_data)} sentences")
    print(f"  Development: {len(dev_data)} sentences")
    print(f"  Combined: {len(combined_train_data)} sentences")
    
    if output_cfg['print_dataset_stats']:
        print_dataset_stats(combined_train_data, "Combined Training Set")
        print_dataset_stats(test_data, "Test Set")
    
    # Train HMM on combined data
    print("\n[2] Training HMM on Combined Data...")
    # Use best configuration: Prefix+Suffix (pref=2, suff=3) - achieves ~90% accuracy
    hmm = HiddenMarkovModel(use_suffix_model=True, suffix_length=3, use_prefix_model=True, prefix_length=2)
    hmm.train(combined_train_data)
    
    if output_cfg['print_hmm_summary']:
        print_hmm_summary(hmm)
    
    # Evaluate on Test Set
    print("\n[3] Evaluating on Test Set...")
    test_results = evaluate_and_report(hmm, test_data, "Test")
    
    # Sample Predictions
    print("\n[4] Sample Predictions...")
    print_sample_predictions(hmm, test_data, n_samples=1, max_tokens=15)
    
    # Save Model
    if training_cfg['save_model']:
        print("\n[5] Saving Model...")
        os.makedirs(training_cfg['model_dir'], exist_ok=True)
        model_path = os.path.join(training_cfg['model_dir'], training_cfg['model_name'])
        hmm.save(model_path)
    
    # Summary
    print("\n" + "="*70)
    print("TRAINING AND EVALUATION COMPLETE!")
    print("="*70)
    print(f"\nFinal Test Accuracy: {test_results['accuracy']:.4f} ({test_results['accuracy']*100:.2f}%)")
    if training_cfg['save_model']:
        print(f"Model saved to: {os.path.join(training_cfg['model_dir'], training_cfg['model_name'])}")
    print("\n" + "="*70)


if __name__ == "__main__":
    import sys
    config_file = sys.argv[1] if len(sys.argv) > 1 else 'config.cfg'
    main(config_file)
