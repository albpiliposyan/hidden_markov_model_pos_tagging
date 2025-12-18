"""Main script for training and evaluating HMM-based POS tagger."""

import os
from hmm import HiddenMarkovModel
from data_utils import load_armenian_dataset, print_dataset_stats
from evaluation import (
    evaluate_and_report, 
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
    
    print(f"  Training: {len(train_data)} sentences")
    print(f"  Development: {len(dev_data)} sentences")
    print(f"  Test: {len(test_data)} sentences")
    
    if output_cfg['print_dataset_stats']:
        print_dataset_stats(train_data, "Training Set")
        print_dataset_stats(dev_data, "Development Set")
        print_dataset_stats(test_data, "Test Set")
    
    # Train multiple HMM configurations to find the best one
    print("\n[2] Training Multiple HMM Configurations...")
    
    configurations = [
        {'name': 'Classical HMM (no affix)', 'params': {'use_suffix_model': False, 'use_prefix_model': False}},
        # {'name': 'Suffix-only (n=2)', 'params': {'use_suffix_model': True, 'suffix_length': 2, 'use_prefix_model': False}},
        {'name': 'Suffix-only (n=3)', 'params': {'use_suffix_model': True, 'suffix_length': 3, 'use_prefix_model': False}},
        # {'name': 'Suffix-only (n=4)', 'params': {'use_suffix_model': True, 'suffix_length': 4, 'use_prefix_model': False}},
        {'name': 'Prefix-only (n=2)', 'params': {'use_suffix_model': False, 'use_prefix_model': True, 'prefix_length': 2}},
        # {'name': 'Prefix+Suffix (pref=2, suff=2)', 'params': {'use_suffix_model': True, 'suffix_length': 2, 'use_prefix_model': True, 'prefix_length': 2}},
        # {'name': 'Prefix+Suffix (pref=2, suff=3)', 'params': {'use_suffix_model': True, 'suffix_length': 3, 'use_prefix_model': True, 'prefix_length': 2}},
        {'name': 'Prefix+Suffix (pref=3, suff=3)', 'params': {'use_suffix_model': True, 'suffix_length': 3, 'use_prefix_model': True, 'prefix_length': 3}}
    ]
    
    dev_results_all = {}
    trained_models = {}
    
    for config in configurations:
        print(f"\n  Training: {config['name']}...")
        hmm = HiddenMarkovModel(**config['params'])
        hmm.train(train_data)
        
        # Evaluate on dev set
        dev_eval = hmm.evaluate(dev_data)
        dev_accuracy = dev_eval['accuracy'] * 100
        dev_results_all[config['name']] = dev_accuracy
        trained_models[config['name']] = hmm
        
        print(f"    Dev Accuracy: {dev_accuracy:.2f}%")
    
    # Select best model based on dev set
    print("\n" + "="*70)
    print("DEVELOPMENT SET ACCURACY SUMMARY")
    print("="*70)
    for name, accuracy in dev_results_all.items():
        print(f"{name:<45} {accuracy:>6.2f}%")
    
    best_model_name = max(dev_results_all, key=dev_results_all.get)
    best_dev_accuracy = dev_results_all[best_model_name]
    hmm = trained_models[best_model_name]
    
    print("\n" + "="*70)
    print(f"BEST MODEL SELECTED: {best_model_name}")
    print(f"Development Accuracy: {best_dev_accuracy:.2f}%")
    print("="*70)
    
    if output_cfg['print_hmm_summary']:
        print("\n" + "="*70)
        print(f"HMM SUMMARY FOR BEST MODEL: {best_model_name}")
        print("="*70)
        print_hmm_summary(hmm)
    
    # Evaluate best model on Development Set
    print("\n[3] Evaluating Best Model on Development Set...")
    dev_results = evaluate_and_report(hmm, dev_data, "Development", method='viterbi')
    
    # Evaluate best model on Test Set
    print("\n[4] Evaluating Best Model on Test Set...")
    test_results = evaluate_and_report(hmm, test_data, "Test", method='viterbi')
    
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
    print(f"\nBest Model: {best_model_name}")
    print(f"Development Accuracy: {dev_results['accuracy']:.4f} ({dev_results['accuracy']*100:.2f}%)")
    print(f"Test Accuracy: {test_results['accuracy']:.4f} ({test_results['accuracy']*100:.2f}%)")
    if training_cfg['save_model']:
        print(f"Model saved to: {os.path.join(training_cfg['model_dir'], training_cfg['model_name'])}")
    print("\n" + "="*70)


if __name__ == "__main__":
    import sys
    config_file = sys.argv[1] if len(sys.argv) > 1 else 'config.cfg'
    main(config_file)
