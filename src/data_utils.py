"""Data utilities for loading and processing CoNLL-U files."""

from conllu import parse
import os


def load_conllu_file(filepath):
    """Load and parse a CoNLL-U file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        sentences = parse(f.read())
    return sentences


def extract_word_pos_pairs(sentences):
    """Extract (word, POS) pairs from parsed CoNLL-U sentences."""
    dataset = []
    
    for sentence in sentences:
        sentence_pairs = []
        for token in sentence:
            # Skip multi-word tokens (they have a range like "7-8" as ID)
            if isinstance(token['id'], int):
                word = token['form']  # The word itself
                pos = token['upos']   # Universal POS tag
                if word and pos:  # Make sure both exist
                    sentence_pairs.append((word, pos))
        
        if sentence_pairs:  # Only add non-empty sentences
            dataset.append(sentence_pairs)
    
    return dataset


def get_vocabulary_and_tagset(dataset):
    """Extract unique words and POS tags from dataset."""
    vocabulary = set()
    tagset = set()
    
    for sentence in dataset:
        for word, pos in sentence:
            vocabulary.add(word)
            tagset.add(pos)
    
    return vocabulary, tagset


def print_dataset_stats(dataset, name="Dataset"):
    """Print statistics about the dataset."""
    vocab, tags = get_vocabulary_and_tagset(dataset)
    total_tokens = sum(len(sent) for sent in dataset)
    
    print(f"\n{name} Statistics:")
    print(f"  Number of sentences: {len(dataset)}")
    print(f"  Total tokens: {total_tokens}")
    print(f"  Vocabulary size: {len(vocab)}")
    print(f"  Number of POS tags: {len(tags)}")
    print(f"  POS tags: {sorted(tags)}")


def load_armenian_dataset(dataset_dir='dataset'):
    """Load Armenian UD dataset and return (train_data, dev_data, test_data)."""
    train_path = os.path.join(dataset_dir, 'hy_armtdp-ud-train.conllu')
    dev_path = os.path.join(dataset_dir, 'hy_armtdp-ud-dev.conllu')
    test_path = os.path.join(dataset_dir, 'hy_armtdp-ud-test.conllu')
    
    print("Loading CoNLL-U files...")
    train_sentences = load_conllu_file(train_path)
    dev_sentences = load_conllu_file(dev_path)
    test_sentences = load_conllu_file(test_path)
    
    print("Extracting word-POS pairs...")
    train_data = extract_word_pos_pairs(train_sentences)
    dev_data = extract_word_pos_pairs(dev_sentences)
    test_data = extract_word_pos_pairs(test_sentences)
    
    return train_data, dev_data, test_data
