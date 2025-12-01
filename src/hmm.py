"""Hidden Markov Model for Part-of-Speech Tagging."""

import numpy as np
import sys


class HiddenMarkovModel:
    """HMM for POS tagging using Maximum Likelihood Estimation."""
    
    def __init__(self, use_suffix_model=True, suffix_length=3):
        """Initialize HMM with default smoothing (1e-10)."""
        self.smoothing = 1e-10
        self.use_suffix_model = use_suffix_model
        self.suffix_length = suffix_length  # Length of suffix for unknown word handling
        self.states = None  # States (POS tags)
        self.vocabulary = None  # Vocabulary (list of unique words)
        self.transition_probs = None  # Transition probabilities: P(tag_j | tag_i)
        self.emission_probs = None  # Emission probabilities: P(word_j | tag_i)
        self.initial_probs = None  # Initial probabilities: P(first tag = tag_i)
        
        # Mappings
        self.tag_to_idx = {}
        self.idx_to_tag = {}
        self.word_to_idx = {}
        self.idx_to_word = {}
        
        # Counts (for analysis)
        self.transition_counts = None
        self.emission_counts = None
        self.tag_counts = None
        self.initial_counts = None
        
        # Suffix-based emission for unknown words
        self.suffix_tag_counts = {}  # suffix -> {tag: count}
        self.suffix_probs = {}  # suffix -> {tag: probability}: P(tag | suffix)
        
        # Marginal tag probabilities P(tag)
        self.tag_marginal_probs = {}  # tag -> P(tag) overall frequency
        
    def train(self, train_data):
        """Train HMM on labeled data (list of sentences with (word, tag) tuples)."""
        print(f"\nTraining HMM with {len(train_data)} sentences...")
        
        # Build vocabulary and tag set
        self._build_vocabulary_and_tagset(train_data)
        
        # Compute probabilities
        self._compute_initial_probabilities(train_data)     # Pi
        self._compute_transition_probabilities(train_data)  # A
        self._compute_emission_probabilities(train_data)    # B
        self._compute_marginal_tag_probabilities(train_data)  # P(tag)
        
        if self.use_suffix_model:
            self._compute_suffix_probabilities(train_data, suffix_length=self.suffix_length)  # Suffix model
        
        print(f"Training complete!")
        print(f"  States (POS tags): {len(self.states)}")
        print(f"  Vocabulary size: {len(self.vocabulary)}")
        if self.use_suffix_model:
            print(f"  Suffix patterns: {len(self.suffix_tag_counts)}")
        
    def _build_vocabulary_and_tagset(self, train_data):
        """Extract unique words and POS tags."""
        Q = set()
        V = set()
        
        for sentence in train_data:
            for word, pos in sentence:
                Q.add(pos)
                V.add(word)
        
        self.states = sorted(list(Q))
        self.vocabulary = sorted(list(V))
        
        # Create mappings
        self.tag_to_idx = {tag: idx for idx, tag in enumerate(self.states)}
        self.idx_to_tag = {idx: tag for tag, idx in self.tag_to_idx.items()}
        self.word_to_idx = {word: idx for idx, word in enumerate(self.vocabulary)}
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}
    
    def _compute_initial_probabilities(self, train_data):
        """Compute Pi[i] = P(first tag = tag_i)."""
        # Count first tags
        self.initial_counts = {}
        for tag in self.states:
            self.initial_counts[tag] = 0
        
        for sentence in train_data:
            if sentence:
                first_tag = sentence[0][1]
                self.initial_counts[first_tag] += 1
        
        total_sentences = len(train_data)
        n_tags = len(self.states)
        self.initial_probs = np.zeros(n_tags)
        
        for tag in self.states:
            count = self.initial_counts[tag]
            tag_idx = self.tag_to_idx[tag]
            self.initial_probs[tag_idx] = (count + self.smoothing) / (total_sentences + self.smoothing * n_tags)
        
        # Assertions
        assert np.isclose(np.sum(self.initial_probs), 1.0), f"Initial probabilities do not sum to 1 (sum={np.sum(self.initial_probs)})"
        assert np.all(self.initial_probs > 0), "All initial probabilities must be > 0 (with smoothing)"
        assert np.all(self.initial_probs <= 1.0), "All initial probabilities must be <= 1.0"
    
    def _compute_transition_probabilities(self, train_data):
        """Compute transition_probs[i][j] = P(tag_j | tag_i)."""
        # Initialize count dictionaries
        self.transition_counts = {}
        self.tag_counts = {}
        
        for tag in self.states:
            self.transition_counts[tag] = {}
            self.tag_counts[tag] = 0
            for next_tag in self.states:
                self.transition_counts[tag][next_tag] = 0
        
        # Count transitions and tags
        for sentence in train_data:
            for i in range(len(sentence) - 1):
                current_tag = sentence[i][1]
                next_tag = sentence[i + 1][1]
                
                self.transition_counts[current_tag][next_tag] += 1
                self.tag_counts[current_tag] += 1
        
        # Compute probabilities with smoothing
        n_tags = len(self.states)
        self.transition_probs = np.zeros((n_tags, n_tags))
        
        for i in range(n_tags):
            tag_i = self.states[i]
            total = self.tag_counts[tag_i]
            
            for j in range(n_tags):
                tag_j = self.states[j]
                count = self.transition_counts[tag_i][tag_j]
                self.transition_probs[i][j] = (count + self.smoothing) / (total + self.smoothing * n_tags)

            # Assertions for each row
            assert np.isclose(np.sum(self.transition_probs[i, :]), 1.0), f"Row {i} of Transition matrix A does not sum to 1"
            assert np.all(self.transition_probs[i, :] > 0), f"Row {i} has zero/negative probabilities (smoothing failed)"
            assert np.all(self.transition_probs[i, :] <= 1.0), f"Row {i} has probabilities > 1.0"
    
    def _compute_emission_probabilities(self, train_data):
        """Compute emission_probs[i][j] = P(word_j | tag_i)."""
        # Initialize count dictionaries
        self.emission_counts = {}
        tag_total_counts = {}
        
        for tag in self.states:
            self.emission_counts[tag] = {}
            tag_total_counts[tag] = 0
            for word in self.vocabulary:
                self.emission_counts[tag][word] = 0
        
        # Count emissions
        for sentence in train_data:
            for word, pos in sentence:
                self.emission_counts[pos][word] += 1
                tag_total_counts[pos] += 1
        
        # Compute probabilities with smoothing
        n_tags = len(self.states)
        n_words = len(self.vocabulary)
        self.emission_probs = np.zeros((n_tags, n_words))
        
        for i in range(len(self.states)):
            tag = self.states[i]
            total = tag_total_counts[tag]
            
            for j in range(len(self.vocabulary)):
                word = self.vocabulary[j]
                count = self.emission_counts[tag][word]
                # Add-epsilon smoothing already normalizes
                self.emission_probs[i][j] = (count + self.smoothing) / (total + self.smoothing * n_words)
            
            # Assertions for each row
            assert np.isclose(np.sum(self.emission_probs[i, :]), 1.0), f"Row {i} of Emission matrix B does not sum to 1"
            assert np.all(self.emission_probs[i, :] > 0), f"Row {i} has zero/negative emission probabilities"
            assert np.all(self.emission_probs[i, :] <= 1.0), f"Row {i} has emission probabilities > 1.0"
    
    def _compute_marginal_tag_probabilities(self, train_data):
        """Compute P(tag) = count(tag) / total_tokens for all tags."""
        tag_counts = {tag: 0 for tag in self.states}
        total_tokens = 0
        
        for sentence in train_data:
            for _, tag in sentence:
                tag_counts[tag] += 1
                total_tokens += 1
        
        # Compute marginal probabilities
        for tag in self.states:
            self.tag_marginal_probs[tag] = tag_counts[tag] / total_tokens if total_tokens > 0 else 0
        
        # Assertions
        assert np.isclose(sum(self.tag_marginal_probs.values()), 1.0), "Marginal tag probabilities do not sum to 1"
    
    def _compute_suffix_probabilities(self, train_data, suffix_length=3):
        """Compute emission probabilities based on word suffixes for unknown words."""
        # Count suffix-tag co-occurrences
        for sentence in train_data:
            for word, tag in sentence:
                if len(word) > suffix_length:
                    suffix = word[-suffix_length:]
                    if suffix not in self.suffix_tag_counts:
                        self.suffix_tag_counts[suffix] = {}
                    self.suffix_tag_counts[suffix][tag] = self.suffix_tag_counts[suffix].get(tag, 0) + 1
        
        # Compute probabilities with smoothing
        n_tags = len(self.states)
        for suffix, tag_counts in self.suffix_tag_counts.items():
            total = sum(tag_counts.values())
            self.suffix_probs[suffix] = {}
            
            for tag in self.states:
                count = tag_counts.get(tag, 0)
                self.suffix_probs[suffix][tag] = (count + self.smoothing) / (total + self.smoothing * n_tags)
        
        # Assertions
        for suffix, probs in self.suffix_probs.items():
            assert np.isclose(sum(probs.values()), 1.0), f"Suffix probabilities for '{suffix}' do not sum to 1"
    
    def _get_unknown_word_emission(self, word, tag):
        """Get emission probability for unknown word using suffix information."""
        if not self.use_suffix_model:
            # Basic uniform probability for unknown words
            return 1e-10
        
        # Try suffix-based probability
        if len(word) > self.suffix_length:
            suffix = word[-self.suffix_length:]
            if suffix in self.suffix_probs:
                return self.suffix_probs[suffix].get(tag, 1e-10)
        
        # Fallback: use scaled marginal probabilities P(tag)
        # Unknown words get probabilities proportional to overall tag frequency
        base_prob = self.tag_marginal_probs.get(tag, 1.0 / len(self.states))

        # Scale down but maintain relative proportions
        return base_prob * 1e-6
    
    def viterbi(self, sentence_words):
        """Viterbi algorithm: find most likely POS tag sequence for sentence."""
        n_states = len(self.states)
        n_obs = len(sentence_words)
        
        # Initialize matrices
        viterbi_matrix = np.zeros((n_states, n_obs))
        backpointer = np.zeros((n_states, n_obs), dtype=int)
        
        # Initialization (t=0)
        for s in range(n_states):
            word = sentence_words[0]
            tag = self.states[s]
            
            if word in self.word_to_idx:
                word_idx = self.word_to_idx[word]
                emission_prob = self.emission_probs[s][word_idx]
            else:
                emission_prob = self._get_unknown_word_emission(word, tag)
            
            viterbi_matrix[s][0] = self.initial_probs[s] * emission_prob
        
        # Recursion (t=1 to T-1)
        for t in range(1, n_obs):
            word = sentence_words[t]
            
            for s in range(n_states):
                tag = self.states[s]
                
                # Get emission probability for current state and word
                if word in self.word_to_idx:
                    word_idx = self.word_to_idx[word]
                    emission_prob = self.emission_probs[s][word_idx]
                else:
                    emission_prob = self._get_unknown_word_emission(word, tag)
                
                # Find max probability path to this state
                max_prob = -1
                max_state = 0
                
                for s_prev in range(n_states):
                    prob = viterbi_matrix[s_prev][t-1] * self.transition_probs[s_prev][s]
                    if prob > max_prob:
                        max_prob = prob
                        max_state = s_prev
                
                # Multiply by emission probability once
                viterbi_matrix[s][t] = max_prob * emission_prob
                backpointer[s][t] = max_state
        
        # Termination
        best_path_prob = -1
        best_last_state = 0
        
        for s in range(n_states):
            if viterbi_matrix[s][n_obs-1] > best_path_prob:
                best_path_prob = viterbi_matrix[s][n_obs-1]
                best_last_state = s
        
        # Backtrack
        best_path = [best_last_state]
        
        for t in range(n_obs-1, 0, -1):
            best_path.insert(0, backpointer[best_path[0]][t])
        
        # Convert to tags
        predicted_tags = [self.idx_to_tag[state] for state in best_path]
        
        return predicted_tags
    
    def predict(self, sentence_words):
        """Alias for viterbi."""
        return self.viterbi(sentence_words)
    
    def evaluate(self, test_data):
        """Evaluate HMM accuracy on test data."""
        total_tokens = 0
        correct_predictions = 0
        
        for sentence in test_data:
            words = [word for word, pos in sentence]
            true_tags = [pos for word, pos in sentence]
            
            predicted_tags = self.predict(words)
            
            for true_tag, pred_tag in zip(true_tags, predicted_tags):
                total_tokens += 1
                if true_tag == pred_tag:
                    correct_predictions += 1
        
        accuracy = correct_predictions / total_tokens if total_tokens > 0 else 0
        
        return {
            'total_tokens': total_tokens,
            'correct': correct_predictions,
            'incorrect': total_tokens - correct_predictions,
            'accuracy': accuracy
        }
    
    def unknown_suffix_statistics(self, test_data):
        """
        Calculate statistics on how many unknown words have suffixes that are also unknown.
        
        Returns:
            dict: Statistics including counts and percentages of unknown words with known/unknown suffixes
        """
        if not self.use_suffix_model:
            print("Suffix model is not enabled.")
            return None
        
        total_unknown_words = 0
        unknown_suffix_count = 0
        known_suffix_count = 0
        too_short_count = 0
        
        for sentence in test_data:
            for word, _ in sentence:
                if word not in self.word_to_idx:
                    total_unknown_words += 1
                    if len(word) > self.suffix_length:
                        suffix = word[-self.suffix_length:]
                        if suffix in self.suffix_probs:
                            known_suffix_count += 1
                        else:
                            unknown_suffix_count += 1
                    else:
                        too_short_count += 1
        
        stats = {
            'total_unknown_words': total_unknown_words,
            'known_suffix_count': known_suffix_count,
            'unknown_suffix_count': unknown_suffix_count,
            'too_short_count': too_short_count,
            'known_suffix_percent': (known_suffix_count / total_unknown_words * 100) if total_unknown_words > 0 else 0.0,
            'unknown_suffix_percent': (unknown_suffix_count / total_unknown_words * 100) if total_unknown_words > 0 else 0.0,
            'too_short_percent': (too_short_count / total_unknown_words * 100) if total_unknown_words > 0 else 0.0
        }
        
        return stats
    
    def get_info(self):
        """Return HMM configuration details."""
        return {
            'n_states': len(self.states),
            'n_words': len(self.vocabulary),
            'states': self.states,
            'smoothing': self.smoothing,
            'transition_matrix_shape': self.transition_probs.shape,
            'emission_matrix_shape': self.emission_probs.shape,
        }
    
    def save(self, filepath):
        """Save the trained HMM."""
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        print(f"Model saved to {filepath}")
    
    @staticmethod
    def load(filepath):
        """Load the trained HMM."""
        import pickle
        with open(filepath, 'rb') as f:
            hmm = pickle.load(f)
        print(f"Model loaded from {filepath}")
        return hmm
