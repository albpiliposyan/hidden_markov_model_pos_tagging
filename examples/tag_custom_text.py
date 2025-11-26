"""Tag custom Armenian text with trained HMM model."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from hmm import HiddenMarkovModel


def tokenize_armenian(text):
    """Simple tokenizer for Armenian text."""
    # Split by whitespace and punctuation
    import re
    # Keep punctuation as separate tokens
    tokens = re.findall(r'\w+|[^\w\s]', text, re.UNICODE)
    return [token for token in tokens if token.strip()]


def tag_text(text, model_path='models/hmm_armenian_pos.pkl'):
    """Tag Armenian text with POS tags."""
    
    # Load trained model
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        print("Please run src/main.py first to train the model.")
        return
    
    hmm = HiddenMarkovModel.load(model_path)
    
    # Split text into lines (treating each line as a sentence)
    lines = [line.strip() for line in text.strip().split('\n') if line.strip()]
    
    print("\n" + "="*70)
    print("POS TAGGING RESULTS")
    print("="*70)
    
    for line_num, line in enumerate(lines, 1):
        print(f"\nLine {line_num}:")
        print(f"Text: {line}")
        print()
        
        # Tokenize
        words = tokenize_armenian(line)
        
        if not words:
            print("  (empty line)")
            continue
        
        # Predict tags
        tags = hmm.predict(words)
        
        # Display results
        print(f"{'№':<4} {'Word':<20} {'POS Tag':<10}")
        print("-" * 40)
        for i, (word, tag) in enumerate(zip(words, tags), 1):
            print(f"{i:<4} {word:<20} {tag:<10}")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    # Your custom text
    armenian_text = """
    Ես իմ անուշ Հայաստանի արևահամ բարն եմ սիրում,
    Մեր հին սազի ողբանվագ, լացակումած լարն եմ սիրում,
    Արնանման ծաղիկների ու վարդերի բույրը վառման,
    Ու նաիրյան աղջիկների հեզաճկուն պա՛րն եմ սիրում։
    """
    
    tag_text(armenian_text)
