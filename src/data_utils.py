"""Data utilities for loading and processing CoNLL-U files."""

from conllu import parse
import os


def load_conllu_file(filepath):
    """Load and parse a .conllu file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        sentences = parse(f.read())
    return sentences


def extract_word_pos_pairs(sentences):
    """Extract (word, POS) pairs from conllu sentences."""
    dataset = []
    
    for sentence in sentences:
        sentence_pairs = []
        for token in sentence:
            # Skip tokens containing multiple words (check the ID as "7-8")
            if isinstance(token['id'], int):
                word = token['form']
                pos = token['upos']
                if word and pos:
                    sentence_pairs.append((word, pos))
        
        if sentence_pairs:
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
    total_words = sum(len(sentence) for sentence in dataset)
    
    print(f"\n{name} Statistics:")
    print(f"  Number of sentences: {len(dataset)}")
    print(f"  Total words: {total_words}")
    print(f"  Vocabulary size: {len(vocab)}")
    print(f"  Number of POS tags: {len(tags)}")
    print(f"  POS tags: {sorted(tags)}")


def load_armenian_dataset(dataset_dir='dataset'):
    """Load dataset and return (train_data, dev_data, test_data)."""
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
