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

The Armenian UD Treebank is divided into three standard splits:

| Split | Sentences | Tokens | Purpose |
|-------|-----------|---------|----------|
| **Training** | 1,974 | 42,069 | Model parameter estimation |
| **Development** | 249 | 5,359 | Hyperparameter tuning |
| **Test** | 277 | 5,157 | Final evaluation |
| **Total** | 2,500 | 52,585 | Complete dataset |

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

### Language Features

Armenian is an Indo-European language with the following relevant characteristics for POS tagging:

- **Rich morphology**: Highly inflected with complex case and verb systems
- **Agglutinative features**: Multiple affixes can be attached to word stems
- **Free word order**: Relatively flexible word order compared to English
- **Pro-drop**: Subject pronouns can be omitted
- **Postpositions**: Uses postpositions in addition to prepositions

### Annotation Quality

- **Manual annotation**: All tags are human-verified
- **Consistency**: Follows Universal Dependencies annotation guidelines
- **Reliability**: Part of the established UD benchmark corpus
- **Transliteration included**: Provides Latin transliteration for accessibility

## Domain and Genre

The dataset comprises texts from various domains:

- **Nonfiction**: Essays, articles, and informative texts
- **Fiction**: Literary works and narratives
- **News**: Journalistic content
- **Web**: Online content and blogs

This diversity ensures the trained model generalizes well across different text types.

## Usage in This Project

For our Hidden Markov Model POS tagger:

1. **Training phase**: Uses word-tag pairs from the training split to estimate:
   - Initial state probabilities (π)
   - Transition probabilities (A) between POS tags
   - Emission probabilities (B) from tags to words

2. **Development phase**: The development set is used for:
   - Validating model performance during development
   - Monitoring for overfitting
   - Comparing different model configurations

3. **Testing phase**: The test set provides:
   - Unbiased final evaluation
   - Performance metrics (accuracy, confusion matrix)
   - Error analysis for model improvement

## Data Loading

The dataset files are loaded using the `conllu` Python library, which parses CoNLL-U format and extracts (word, POS tag) pairs for training. Only the FORM (column 2) and UPOS (column 4) fields are utilized in this implementation, as the HMM focuses exclusively on word-to-tag relationships.

## Challenges

Specific challenges this dataset presents for HMM-based POS tagging:

1. **Out-of-vocabulary (OOV) words**: The test set contains words not seen during training
2. **Ambiguous words**: Many Armenian words can have multiple POS tags depending on context
3. **Rare tags**: Some tags (SYM, INTJ, X) have very few examples
4. **Morphological complexity**: Rich inflection creates large vocabulary with sparse observations
5. **Free word order**: Syntactic flexibility reduces the predictive power of tag sequences

Our implementation addresses these through add-epsilon smoothing (ε = 1×10⁻¹⁰) to handle unseen events and proper probability estimation techniques.

## References

- Universal Dependencies project: https://universaldependencies.org/
- Armenian UD Treebank: https://github.com/UniversalDependencies/UD_Armenian-ArmTDP
- CoNLL-U format specification: https://universaldependencies.org/format.html
