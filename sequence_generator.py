from enum import Enum, auto

import torch

from model import autoregressive_mask


class DecodingStrategy(Enum):
    """
    Enum class for different decoding strategies.
    """
    TOP_K = auto()
    TOP_P = auto()
    GREEDY = auto()
    RANDOM = auto()
    BEAM_SEARCH = auto()


class SequenceGenerator:
    def __init__(self, model, sos_token, eos_token, pad_token, max_len=50):
        """
        Initializes the sequence generator with a model and parameters for decoding.
        Args:
            model (torch.nn.Module): The trained transformer for generating predictions.
            sos_token (int): The index of the start symbol in the vocabulary.
            eos_token (int): The index of the end symbol in the vocabulary.
            pad_token (int): The index of the padding symbol in the vocabulary.
            max_len (int): The maximum length of the output sequence to generate.
        """
        self.model = model
        self.sos_token = sos_token
        self.eos_token = eos_token
        self.pad_token = pad_token
        self.max_len = max_len

    def generate(self, src, src_mask, strategy=DecodingStrategy.GREEDY, k=None, p=None):
        """
        Performs batched autoregressive generation on the model's output using different sampling techniques.
        Args:
            src (torch.Tensor): The encoded source sequence tensor. Shape: [batch_size, seq_len, feature_dim]
            src_mask (torch.Tensor): The mask tensor for the source sequence. Shape: [batch_size, 1, seq_len]
            strategy (DecodingStrategy): The decoding strategy to use. Defaults to DecodingStrategy.GREEDY.
        Returns:
            List[List[int]]: A batch of decoded sequences of tokens.
        """
        batch_size = src.size(0)
        out_tokens = torch.full((batch_size, 1), self.sos_token, dtype=torch.long).to(src.device)
        for i in range(self.max_len - 1):  # -1 to account for the SOS token
            prob = None
            # YOUR CODE STARTS HERE - do a forward pass through the model to get decoder output
            # Project the decoder output to generator for next token probabilities
            # Hint: use autoregressive_mask to create the tgt_mask
            # Hint: remember that generator gives log_probs, but sampling requires probs
            # TODO: Implement the functionality to get the next token probabilities

            # YOUR CODE ENDS HERE
            # These are the different decoding strategies to generate the next token
            # Will be implemented in the following methods
            if strategy == DecodingStrategy.GREEDY:
                next_word, next_word_log_prob = self.sample_greedy(prob)
            elif strategy == DecodingStrategy.RANDOM:
                next_word, next_word_log_prob = self.sample_random(prob)
            elif strategy == DecodingStrategy.TOP_K:
                next_word, next_word_log_prob = self.sample_top_k(prob, k=k)
            elif strategy == DecodingStrategy.TOP_P:
                next_word, next_word_log_prob = self.sample_top_p(prob, p=p)
            else:
                raise ValueError(f"Invalid decoding strategy: {strategy}")
            # TODO: Implement the functionality to append the next_word to the out_tokens tensor
            # YOUR CODE STARTS HERE
            # append the next_word to the ys tensor
            # break the loop if all sequences have generated the EOS token (important for efficiency)

            # YOUR CODE ENDS HERE

        # Remove sequences after the end symbol for each batch item
        decoded_sequences = []
        # TODO: Implement the functionality to remove tokens after the EOS token
        # YOUR CODE STARTS HERE
        # for each sequence in the batch, remove the padding tokens and append the sequence to the decoded_sequences
        # list (remember to convert to list of ints)

        # YOUR CODE ENDS HERE
        return decoded_sequences

    def beam_search(self, src, src_mask, beam_width=3):
        """
          Perform beam search decoding for a single input sequence.
          Args:
              src (torch.Tensor): The encoded source sequence tensor. Shape: [1, seq_len, feature_dim]
              src_mask (torch.Tensor): The mask tensor for the source sequence. Shape: [1, 1, seq_len]
              beam_width (int): The number of sequences to keep at each step in the beam.
          Returns:
              List[int]: The best sequence of token IDs based on beam search.
      """
        batch_size = src.size(0)
        assert batch_size == 1, "Beam search is implemented for a single sequence only."

        # Starting with the initial token.
        ys = torch.full((1, 1), self.sos_token, dtype=torch.long).to(src.device)
        beam_candidates = [(ys, 0)]  # list of tuples (sequence tensor, log probability)

        for _ in range(self.max_len - 1):  # -1 for the sos token
            all_candidates = []
            for ys, log_prob in beam_candidates:
                # TODO: Implement the functionality to get the log probabilities of the next token using the model's decode method
                # YOUR CODE STARTS HERE
                """
                  Steps:
                  1. Get the log probabilities of the next token using the model's decode method.
                  2. Get the top beam_width tokens (by probability values) and their log probabilities.
                  3. Create new candidate sequences by appending each of the top tokens to the current sequence.
                  4. Add the new candidate sequences to the list of all candidates.
                  HINT: The idea will be similar to generate, but you will have to keep track of multiple sequences.
                """

                # YOUR CODE ENDS HERE

            # TODO: Implement the functionality to sort all candidates by log probability and select the best beam_width ones
            # YOUR CODE STARTS HERE - Sort all candidates by log probability, select the best beam_width ones
            # Sort all candidates by log probability, select the best beam_width ones

            # YOUR CODE ENDS HERE

            # Check if the end token is generated and stop early
            if all((c[0][0, -1] == self.eos_token) for c in beam_candidates):
                break

        # Choose the sequence with the highest log probability
        best_sequence, _ = max(beam_candidates, key=lambda x: x[1])
        result = best_sequence[0].tolist()
        return result

    @staticmethod
    def sample_greedy(prob):
        """
        Perform greedy decoding to get the next token index based on the probability distribution.
        Steps -
        1. Get the index of the token with the highest probability.
        2. Retrieve the log probability of the chosen token
        Args:
            prob (torch.Tensor): The probability distribution over the target vocabulary of shape
            [batch_size, vocab_size].

        Returns:
            torch.Tensor: The index of the next token of shape [batch_size].
            torch.Tensor: The log probability of the chosen token of shape [batch_size].
        HINTS:
        - The functions torch.gather may be useful.
        """
        next_word, log_probability_of_next_word = None, None
        # TODO: Implement Greedy Sampling
        # YOUR CODE STARTS HERE

        # YOUR CODE ENDS HERE
        return next_word, log_probability_of_next_word

    @staticmethod
    def sample_random(prob):
        """
        Perform random sampling to get the next token index based on the probability distribution.
        Steps -
        1. Sample from the probability distribution over the target vocabulary.
        2. Retrieve the log probability of the chosen token.
        3. Map sampled indices back to the global vocabulary indices.
        Args:
            prob (torch.Tensor): The probability distribution of the batch over the target vocabulary.
        Returns:
            torch.Tensor: The index of the next token of shape [batch_size].
            torch.Tensor: The log probability of the chosen token of shape [batch_size].
        HINTS:
        - The functions torch.multinomial and torch.gather may be useful.
        """
        next_word, log_probability_of_next_word = None, None
        # TODO: Implement Random Sampling
        # YOUR CODE STARTS HERE

        # YOUR CODE ENDS HERE
        return next_word, log_probability_of_next_word

    @staticmethod
    def sample_top_k(prob, k=5):
        """
        Perform top-k sampling to get the next token index based on the probability distribution.
        Steps -
        1. Filter the top k tokens from the distribution.
        2. Normalize the probabilities to sum to 1.
        3. Randomly sample from this modified distribution of top-k tokens to determine the next token.
        4. Retrieve the log probability and index of the chosen token in the global vocabulary.
        Args:
            prob (torch.Tensor): The probability distribution of the batch over the target vocabulary.
            k (int): The number of top elements to sample from.
        Returns:
            torch.Tensor: The index of the next token of shape [batch_size].
            torch.Tensor: The log probability of the chosen token of shape [batch_size].
        HINTS -
        - The function torch.topk may be useful.
        """
        next_word, log_probability_of_next_word = None, None
        # TODO: Implement Top-k Sampling
        # YOUR CODE STARTS HERE

        # YOUR CODE ENDS HERE
        return next_word, log_probability_of_next_word

    @staticmethod
    def sample_top_p(prob, p=0.9):
        """
        Perform top-p sampling to get the next token index based on the probability distribution.
        Steps -
        1. Retrieve the smallest subset of the distribution that sums just greater than p
        (since = isn't always possible).
        2. Normalize the probabilities to sum to 1.
        3. Randomly sample from this modified distribution to determine the next token.
        4. Retrieve the log probability and index of the chosen token in the global vocabulary.
        Args:
            prob (torch.Tensor): The probability distribution of the batch over the target vocabulary.
            p (float): The cumulative probability threshold for top-p sampling.
        Returns:
            torch.Tensor: The index of the next token of shape [batch_size].
            torch.Tensor: The log probability of the chosen token of shape [batch_size]
        HINTS:
        - The function torch.cumsum may be useful.
        """
        next_word, log_probability_of_next_word = None, None
        # TODO: Implement Top-p Sampling
        # YOUR CODE STARTS HERE

        # YOUR CODE ENDS HERE
        return next_word, log_probability_of_next_word
