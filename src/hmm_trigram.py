"""Second-order (Trigram) Hidden Markov Model for Part-of-Speech Tagging."""

import numpy as np


class TrigramHMM:
    """Second-order HMM where next tag depends on previous two tags."""
    
    def __init__(self):
        """Initialize trigram HMM with default smoothing (1e-10)."""
        self.smoothing = 1e-10
        self.tags = None  # List of individual POS tags
        self.states = None  # List of state pairs (tag_i, tag_j)
        self.V = None  # Vocabulary
        self.A = None  # Transition probabilities: P(tag_k | tag_i, tag_j)
        self.B = None  # Emission probabilities: P(word | tag_i, tag_j)
        self.pi = None  # Initial probabilities for first two tags
        
        # Mappings
        self.tag_to_idx = {}
        self.state_to_idx = {}
        self.idx_to_state = {}
        self.word_to_idx = {}
        self.idx_to_word = {}
        
        # Counts (for analysis)
        self.transition_counts = None
        self.emission_counts = None
        self.state_counts = None
        self.initial_counts = None
        
    def train(self, train_data):
        """Train trigram HMM on labeled data."""
        print(f"\nTraining Trigram HMM with {len(train_data)} sentences...")
        
        # Build vocabulary and tag set
        self._build_vocabulary_and_tagset(train_data)
        
        # Create state space (pairs of tags)
        self._build_state_space()
        
        # Compute probabilities
        self._compute_initial_probabilities(train_data)
        self._compute_transition_probabilities(train_data)
        self._compute_emission_probabilities(train_data)
        
        print(f"Training complete!")
        print(f"  Individual POS tags: {len(self.tags)}")
        print(f"  States (tag pairs): {len(self.states)}")
        print(f"  Vocabulary size: {len(self.V)}")
        
    def _build_vocabulary_and_tagset(self, train_data):
        """Extract unique words and POS tags."""
        tags_set = set()
        words_set = set()
        
        for sentence in train_data:
            for word, pos in sentence:
                tags_set.add(pos)
                words_set.add(word)
        
        self.tags = sorted(list(tags_set))
        self.V = sorted(list(words_set))
        
        # Create tag mappings
        self.tag_to_idx = {tag: idx for idx, tag in enumerate(self.tags)}
        self.word_to_idx = {word: idx for idx, word in enumerate(self.V)}
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}
    
    def _build_state_space(self):
        """Create all possible pairs of tags as states."""
        self.states = []
        for tag1 in self.tags:
            for tag2 in self.tags:
                self.states.append((tag1, tag2))
        
        self.state_to_idx = {state: idx for idx, state in enumerate(self.states)}
        self.idx_to_state = {idx: state for state, idx in self.state_to_idx.items()}
    
    def _compute_initial_probabilities(self, train_data):
        """Compute P(tag1, tag2 for first two positions)."""
        self.initial_counts = {}
        for state in self.states:
            self.initial_counts[state] = 0
        
        # Count first two tags in sentences
        for sentence in train_data:
            if len(sentence) >= 2:
                first_tag = sentence[0][1]
                second_tag = sentence[1][1]
                state = (first_tag, second_tag)
                self.initial_counts[state] += 1
        
        total_sentences = len(train_data)
        n_states = len(self.states)
        self.pi = np.zeros(n_states)
        
        for state in self.states:
            count = self.initial_counts[state]
            state_idx = self.state_to_idx[state]
            # Add-epsilon smoothing
            self.pi[state_idx] = (count + self.smoothing) / (total_sentences + self.smoothing * n_states)
    
    def _compute_transition_probabilities(self, train_data):
        """Compute A[i][j] = P(tag_k | tag_i, tag_j) where state is (tag_i, tag_j)."""
        # Initialize counts
        self.transition_counts = {}
        self.state_counts = {}
        
        for state in self.states:
            self.transition_counts[state] = {}
            self.state_counts[state] = 0
            for next_tag in self.tags:
                self.transition_counts[state][next_tag] = 0
        
        # Count trigrams
        for sentence in train_data:
            for i in range(len(sentence) - 2):
                tag1 = sentence[i][1]
                tag2 = sentence[i + 1][1]
                tag3 = sentence[i + 2][1]
                
                state = (tag1, tag2)
                self.transition_counts[state][tag3] += 1
                self.state_counts[state] += 1
            
            # Count last state (for proper normalization)
            if len(sentence) >= 2:
                tag1 = sentence[-2][1]
                tag2 = sentence[-1][1]
                state = (tag1, tag2)
                self.state_counts[state] += 1
        
        # Compute probabilities
        n_states = len(self.states)
        n_tags = len(self.tags)
        self.A = np.zeros((n_states, n_tags))
        
        for state_idx, state in enumerate(self.states):
            total = self.state_counts[state]
            
            for tag_idx, next_tag in enumerate(self.tags):
                count = self.transition_counts[state][next_tag]
                # Add-epsilon smoothing
                self.A[state_idx][tag_idx] = (count + self.smoothing) / (total + self.smoothing * n_tags)
    
    def _compute_emission_probabilities(self, train_data):
        """Compute B[i][j] = P(word_j | state_i) where state is (tag1, tag2)."""
        # Initialize counts
        self.emission_counts = {}
        state_total_counts = {}
        
        for state in self.states:
            self.emission_counts[state] = {}
            state_total_counts[state] = 0
            for word in self.V:
                self.emission_counts[state][word] = 0
        
        # Count emissions from second tag in each state
        for sentence in train_data:
            for i in range(len(sentence) - 1):
                tag1 = sentence[i][1]
                tag2 = sentence[i + 1][1]
                word = sentence[i + 1][0]
                
                state = (tag1, tag2)
                self.emission_counts[state][word] += 1
                state_total_counts[state] += 1
        
        # Compute probabilities
        n_states = len(self.states)
        n_words = len(self.V)
        self.B = np.zeros((n_states, n_words))
        
        for state_idx, state in enumerate(self.states):
            total = state_total_counts[state]
            
            for word_idx, word in enumerate(self.V):
                count = self.emission_counts[state][word]
                # Add-epsilon smoothing
                self.B[state_idx][word_idx] = (count + self.smoothing) / (total + self.smoothing * n_words)
    
    def viterbi(self, sentence_words):
        """Viterbi algorithm for second-order HMM."""
        if len(sentence_words) < 2:
            # Fallback for very short sentences
            return [self.tags[0]] * len(sentence_words)
        
        n_states = len(self.states)
        n_obs = len(sentence_words)
        
        # Handle unknown words
        unk_prob = 1e-10
        
        # Initialize matrices - states represent (tag_{t-1}, tag_t)
        viterbi_matrix = np.zeros((n_states, n_obs))
        backpointer = np.zeros((n_states, n_obs), dtype=int)
        
        # Initialization (t=1) - position 1 (second word)
        # State (tag_0, tag_1) represents tags at positions 0 and 1
        for state_idx, state in enumerate(self.states):
            # Get emission probability for word at position 1
            word = sentence_words[1]
            if word in self.word_to_idx:
                word_idx = self.word_to_idx[word]
                emission_prob = self.B[state_idx][word_idx]
            else:
                emission_prob = unk_prob
            
            # Initial probability * emission
            viterbi_matrix[state_idx][1] = self.pi[state_idx] * emission_prob
        
        # Recursion (t=2 to T-1)
        for t in range(2, n_obs):
            word = sentence_words[t]
            
            # For each possible current state (tag_{t-1}, tag_t)
            for curr_state_idx, curr_state in enumerate(self.states):
                tag_prev, tag_curr = curr_state
                
                # Get emission probability for this state
                if word in self.word_to_idx:
                    word_idx = self.word_to_idx[word]
                    emission_prob = self.B[curr_state_idx][word_idx]
                else:
                    emission_prob = unk_prob
                
                max_prob = -1
                max_prev_state_idx = 0
                
                # For each possible previous state (tag_{t-2}, tag_{t-1})
                # The previous state's second tag must match current state's first tag
                for prev_state_idx, prev_state in enumerate(self.states):
                    tag_prevprev, tag_prev_check = prev_state
                    
                    # Check if states are compatible: prev_state[1] == curr_state[0]
                    if tag_prev_check != tag_prev:
                        continue
                    
                    # Get transition probability P(tag_curr | tag_prevprev, tag_prev)
                    tag_curr_idx = self.tag_to_idx[tag_curr]
                    transition_prob = self.A[prev_state_idx][tag_curr_idx]
                    
                    # Calculate probability
                    prob = viterbi_matrix[prev_state_idx][t-1] * transition_prob
                    
                    if prob > max_prob:
                        max_prob = prob
                        max_prev_state_idx = prev_state_idx
                
                # Store results
                viterbi_matrix[curr_state_idx][t] = max_prob * emission_prob
                backpointer[curr_state_idx][t] = max_prev_state_idx
        
        # Termination - find best final state
        best_path_prob = -1
        best_last_state = 0
        
        for state_idx in range(n_states):
            if viterbi_matrix[state_idx][n_obs-1] > best_path_prob:
                best_path_prob = viterbi_matrix[state_idx][n_obs-1]
                best_last_state = state_idx
        
        # Backtrack to find best path
        path_states = [best_last_state]
        for t in range(n_obs-1, 1, -1):
            path_states.insert(0, backpointer[path_states[0]][t])
        
        # Extract tags from states
        predicted_tags = []
        
        # First tag from first state
        first_state = self.idx_to_state[path_states[0]]
        predicted_tags.append(first_state[0])
        
        # Second tag from first state
        predicted_tags.append(first_state[1])
        
        # Remaining tags: take second tag from each subsequent state
        for i in range(1, len(path_states)):
            state = self.idx_to_state[path_states[i]]
            predicted_tags.append(state[1])
        
        return predicted_tags
    
    def predict(self, sentence_words):
        """Alias for viterbi."""
        return self.viterbi(sentence_words)
    
    def evaluate(self, test_data):
        """Evaluate HMM accuracy on test data."""
        total_tokens = 0
        correct_predictions = 0
        
        for sentence in test_data:
            if len(sentence) < 2:
                continue
                
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
    
    def save(self, filepath):
        """Save the trained trigram HMM."""
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        print(f"Trigram model saved to {filepath}")
    
    @staticmethod
    def load(filepath):
        """Load the trained trigram HMM."""
        import pickle
        with open(filepath, 'rb') as f:
            hmm = pickle.load(f)
        print(f"Trigram model loaded from {filepath}")
        return hmm
