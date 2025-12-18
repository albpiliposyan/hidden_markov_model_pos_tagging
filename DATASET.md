# Dataset Description

## Overview

This project utilizes the **Armenian Treebank (ArmTDP)** from the Universal Dependencies (UD) project. The dataset consists of annotated Armenian text with part-of-speech (POS) tags following the Universal POS tagset standard.

## Dataset Source

- **Name**: Armenian Treebank (hy_armtdp)
- **Framework**: Universal Dependencies v2.x
- **Language**: Armenian (Eastern Armenian)
- **Format**: CoNLL-U (Conference on Computational Natural Language Learning - Universal)
- **License**: Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)
- **Citation**: Marat M. Yavrumyan, Hrant H. Khachatrian, Anna S. Danielyan, Gor D. Arakelyan. "ArMor: Armenian Morphological Annotator." In Proceedings of the Eleventh International Conference on Language Resources and Evaluation (LREC 2018).

## Dataset Statistics

The Armenian UD Treebank contains 2,500 sentences with 52,585 tokens total. This project uses a random 80-10-10 split (with seed=42 for reproducibility):

| Split           | Sentences | Tokens (approx) | Purpose                                    |
|-------|---------|-----------|-----------------|--------------------------------------------|
| **Training**    | ~2,000    | ~42,000         | Model parameter estimation                 |
| **Development** | ~250      | ~5,300          | Hyperparameter tuning and model selection  |
| **Test**        | ~250      | ~5,300          | Final evaluation                           |
| **Total**       | 2,500     | 52,585          | Complete dataset                           |

**Note**: The dataset is randomly shuffled and split at runtime, not using the original predefined splits from the CoNLL-U files. This ensures balanced distribution across splits.

## POS Tag Distribution

The dataset uses 17 Universal POS tags:

1. **NOUN** - Nouns (common and proper in base form)
2. **VERB** - Verbs (all forms)
3. **ADJ** - Adjectives
4. **ADP** - Adpositions (prepositions and postpositions)
5. **DET** - Determiners
6. **PRON** - Pronouns
7. **AUX** - Auxiliary verbs
8. **PROPN** - Proper nouns
9. **ADV** - Adverbs
10. **CCONJ** - Coordinating conjunctions
11. **SCONJ** - Subordinating conjunctions
12. **PART** - Particles
13. **NUM** - Numerals
14. **PUNCT** - Punctuation
15. **INTJ** - Interjections
16. **SYM** - Symbols
17. **X** - Other/Unknown

The most frequent tags in the training set are NOUN, VERB, PUNCT, ADP, and DET, which together constitute the majority of tokens.

## Data Format (CoNLL-U)

Each file follows the CoNLL-U format with 10 tab-separated columns per token:

1. ID - Token counter (starting at 1 for each sentence)
2. FORM - Word form or punctuation symbol
3. LEMMA - Lemma or stem of word form
4. UPOS - Universal part-of-speech tag (used in this project)
5. XPOS - Language-specific part-of-speech tag
6. FEATS - Morphological features
7. HEAD - Head of the current token
8. DEPREL - Dependency relation to the HEAD
9. DEPS - Enhanced dependency graph
10. MISC - Any other annotation

**Example annotation:**
```
# sent_id = nonfiction-006U-00010CQG
# text = Մտածում եմ՝ Ադամի ու Եվայի վտարումը...
1	Մտածում	մտածել	VERB	_	Aspect=Imp|Subcat=Tran|VerbForm=Part|Voice=Act	0	root	_	Translit=Mtaçowm
2	եմ	եմ	AUX	_	Aspect=Imp|Mood=Ind|Number=Sing|Person=1|Polarity=Pos	1	aux	_	Translit=em
```

## Dataset Characteristics

### Annotation Quality

- **Manual annotation**: All tags are human-verified
- **Consistency**: Follows Universal Dependencies annotation guidelines
- **Reliability**: Part of the established UD benchmark corpus
- **Transliteration included**: Provides Latin transliteration for accessibility

## Usage in This Project

For our Hidden Markov Model POS tagger:

1. **Data preprocessing**:
   - All words are converted to lowercase for vocabulary reduction
   - Dataset is randomly shuffled and split (80-10-10) with seed=42
   - Only FORM (word) and UPOS (POS tag) fields are extracted

2. **Training phase**: Uses word-tag pairs from the training split to estimate:
   - Initial state probabilities (π)
   - Transition probabilities (A) between POS tags
   - Emission probabilities (B) from tags to words
   - Suffix/prefix patterns for unknown word handling (enhanced models)

3. **Development phase**: The development set is used for:
   - Model selection among different configurations
   - Hyperparameter validation
   - Monitoring for overfitting

4. **Testing phase**: The test set provides:
   - Unbiased final evaluation
   - Performance metrics (accuracy, confusion matrix)
   - Known vs unknown word analysis

## Data Loading

The dataset files are loaded using the `conllu` Python library, which parses CoNLL-U format. The loading process:

1. Loads all three CoNLL-U files (train, dev, test)
2. Extracts (word, POS tag) pairs from FORM and UPOS columns
3. Converts all words to lowercase
4. Combines all sentences into a single pool
5. Randomly shuffles with seed=42
6. Splits into 80% train, 10% dev, 10% test

This approach ensures reproducible results while maintaining balanced splits across different text domains.

## Challenges and Solutions

Specific challenges this dataset presents for HMM-based POS tagging:

1. **Out-of-vocabulary (OOV) words**: The test set contains words not seen during training
   - **Solution**: Suffix/prefix-enhanced models that learn morphological patterns
   
2. **Ambiguous words**: Many Armenian words can have multiple POS tags depending on context
   - **Solution**: Transition probabilities capture contextual patterns
   
3. **Rare tags**: Some tags (SYM, INTJ, X) have very few examples
   - **Solution**: Add-epsilon smoothing (ε = 1×10⁻¹⁰) handles unseen events
   
4. **Morphological complexity**: Rich inflection creates large vocabulary with sparse observations
   - **Solution**: Lowercase normalization reduces vocabulary size
   
5. **Free word order**: Syntactic flexibility reduces the predictive power of tag sequences
   - **Solution**: Trigram HMM (second-order) captures longer context

## Model Performance

With the random split approach and lowercase normalization, the models achieve:

- **Classical Bigram HMM**: 86.40% accuracy
- **Suffix-enhanced (n=2)**: 91.76% accuracy
- **Suffix-enhanced (n=3)**: **92.06% accuracy** (best model)
- **Prefix-enhanced (n=2)**: 88.68% accuracy
- **Prefix-enhanced (n=3)**: 88.68% accuracy
- **Prefix+Suffix (pref=2, suff=3)**: 91.39% accuracy
- **Trigram HMM**: 71.10% accuracy (suffers from data sparsity)

## References

- Universal Dependencies project: https://universaldependencies.org/
- Armenian UD Treebank: https://github.com/UniversalDependencies/UD_Armenian-ArmTDP
- CoNLL-U format specification: https://universaldependencies.org/format.html
