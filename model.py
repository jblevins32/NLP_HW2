import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy


def clones(module, count):
    """
    Produce N identical layers. This is helpful when you need multiple layers that are identical.
    Args:
        module: the layer to be cloned
        count: number of clones

    Returns: a ModuleList containing N identical layers
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(count)])


def attention(query, key, value, mask=None, dropout=None):
    """
    Compute the Scaled Dot Product Attention.
    The function computes the attention scores based on queries and keys,
    and then scales by the inverse square root of the dimension of the keys. It then applies a softmax to obtain
    weights on the values.
    Apply the mask to the attention weights (optional) and then apply dropout (optional).
    Parameters:
        query (Tensor): Query tensor of shape [batch_size, n_heads, seq_len, d_k]
        key (Tensor): Key tensor of shape [batch_size, n_heads, seq_len, d_q]
        value (Tensor): Value tensor of shape [batch_size, n_heads, seq_len, d_v]
        mask (Tensor, optional): Mask tensor of shape [batch_size, 1, seq_len, seq_len]
            The mask blocks certain positions from being attended to (e.g., padding). Defaults to None.
            Same mask is applied across all attention heads.
        dropout (nn.Module, optional): Dropout module to apply to the attention weights. Defaults to None.
    Returns:
        Tuple[Tensor, Tensor]:
            - The first tensor is the weighted sum of the values, shape [batch_size, n_heads, seq_len, d_v].
            - The second tensor is the attention weights, shape [batch_size, n_heads, seq_len, seq_len].
    NOTE: Remember to handle the mask properly, i.e., the tokens in the mask shouldn't be attended to.
    """
    attention_weights = None
    attented_values = None
    # TODO: Implement attention mechanism
    # YOUR CODE STARTS HERE

    scores = torch.matmul(query, key.transpose(-2,-1)) / math.sqrt(key.size(-1))

    # If mask is provided, apply it to the scores
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    attention_weights = F.softmax(scores, dim=-1)

    # If dropout is provided, apply it to the attention weights
    if dropout is not None:
        attention_weights = dropout(attention_weights)
    attented_values = torch.matmul(attention_weights, value)

    # YOUR CODE ENDS HERE
    return attented_values, attention_weights


def autoregressive_mask(size):
    """
    Creates a mask to prevent a position from attending to subsequent positions in the sequence.
    This ensures that the predictions for position i can depend only on the known outputs at positions less than i.
    Parameters:
        size (int): The size of the last two dimensions of the square attention mask. This should
                    be equal to the length of the sequence.
    Returns:
        torch.Tensor: A 3D tensor of shape (1, size, size) where the value at position (1, i, j)
                      is False if i < j (i.e., mask out position j for position i) and True otherwise.
    """
    # Create a 3D tensor that has 'size' rows and 'size' columns, initialized to 1s

    # TODO: Implement autoregressive mask

    # YOUR CODE STARTS HERE

    # Create mask
    result = torch.ones(1, size, size, dtype=torch.bool)

    # Fill the upper triangular part of the mask with False (where i<j)
    result = torch.triu(result, diagonal=1)

    # YOUR CODE ENDS HERE

    return (result == False).to(torch.bool)


class PositionalEncoding(nn.Module):
    """
    A module to add positional encodings to the input tensor.
    This module computes sinusoidal positional encodings as described in
    "Attention is All You Need" (Vaswani et al., 2017). Positional encodings are
    added to provide the model with information about the relative or absolute
    position of the tokens in the sequence. The encodings themselves as per equations in the notebook.
    Attributes:
            dropout (nn.Dropout): Dropout layer to apply after adding the positional encodings.
            pe (Tensor): A buffer that stores the positional encodings for `max_len` positions.
                         This buffer is not a trainable parameter but is part of the model's
                         state and is registered as a buffer in PyTorch.
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        """
        Initializes the PositionalEncoding module.
        Args:
            d_model (int): The dimensionality of the input embeddings. The same dimensionality
                           is used for the positional encodings to facilitate their addition
                           to the embeddings.
            dropout (float): The dropout rate to apply after adding the positional encodings
                             to the embeddings.
            max_len (int): The maximum length of the input sequences for which positional
                           encodings are precomputed. Defaults to 5000.

        Note: use the `register_buffer` method to register the positional encodings as buffer.
        Use nn.Dropout for dropout.
        """
        super(PositionalEncoding, self).__init__()
        # TODO: Define the positional encoding class initialization
        # YOUR CODE STARTS HERE

        self.dropout = nn.Dropout(p=dropout)

        # Base positional encoding matrix
        pe = torch.zeros(max_len, d_model)

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        divide = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # Replace even and odd indices with sine and cosine, respectively
        pe[:, 0::2] = torch.sin(position * divide)
        pe[:, 1::2] = torch.cos(position * divide)

        # Add a batch dimension to the positional encoding
        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)
        # YOUR CODE ENDS HERE

    def forward(self, x):
        """
        Forward pass of the PositionalEncoding module.
        Adds the positional encodings to the input tensor and applies dropout.
        Parameters:
            x (Tensor): The input tensor to which the positional encodings will be added.
                        Expected shape [batch_size, sequence_length, d_model].
        Returns:
            Tensor: The processed tensor with positional encodings added and dropout applied.
                    Shape [batch_size, sequence_length, d_model].
        Note:
            The `requires_grad_` should be set to False in order to prevent the gradient computations for
            the positional encodings during backpropagation. Since the positional encodings
            are not parameters to be learned but fixed values derived from their position
            in the sequence, they should not affect the gradient flow.
            Apply dropout after adding the positional encodings.
        """
        # Prevent gradient computations for positional encodings

        # TODO: Implement positional encoding forward pass
        # YOUR CODE STARTS HERE 

        out = x + self.pe[:, :x.size(1), :]
        out = self.dropout(out)

        # YOUR CODE ENDS HERE
        return out.detach()


class Embeddings(nn.Module):
    """
    A module to convert token indices to embeddings, scaling by the square root of the model dimension.
    This embedding layer is a common component in models dealing with text data, providing a way to
    project token indices into a continuous and learnable vector space. This is same as previous homeworks.
    Attributes:
        d_model: model dimensionality
        vocab_size: maximum number of tokens in the vocabulary
        embed (nn.Embedding): Embedding layer that converts token indices to vectors
    """

    def __init__(self, d_model, vocab_size):
        """
        Initializes the Embeddings module.
        Args:
            d_model (int): model dimensionality
            vocab_size (int): maximum number of tokens in the vocabulary
        """
        super(Embeddings, self).__init__()
        # TODO: Define embedding layer class initialization
        # YOUR CODE STARTS HERE

        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, d_model)

        # YOUR CODE ENDS HERE

    def forward(self, x):
        """
        The forward pass of the Embeddings module.
        Converts input token indices into embeddings, scales the embeddings by the square root of the
        dimensionality of the model to maintain the variance.
        Parameters:
            x (Tensor): The input tensor of token indices. Expected shape [batch_size, sequence_length].

        Returns:
            Tensor: The scaled embeddings tensor. Shape [batch_size, sequence_length, d_model].
        """
        out = None
        # TODO: Implement embedding forward pass
        # YOUR CODE STARTS HERE

        out = self.embed(x) * math.sqrt(self.d_model)

        # YOUR CODE ENDS HERE
        return out 

    def set_embedding_weights(self):
        """
        Set the weights of the embedding layer.
        """
        for idx in range(self.vocab_size):
            self.embed.weight.data[idx] = torch.linspace(start=0.0, end=1.0, steps=self.d_model)


class MultiHeadedAttention(nn.Module):
    """
    Implements a Multi-Headed Attention mechanism as described in the transformer model architecture.
    Each head computes an independent attention, and their outputs are concatenated and linearly transformed
    into the expected dimensionality.
    Attributes:
            d_k (int): Dimension of each key/query/value vector per head.
            h (int): Number of heads.
            d_model (int): Total dimension of the model.
            linear layers (ModuleList): A list of linear layers for projections.
            Utilize the `clones` function to create multiple instances of a module.
            The linear layers are stored in the following order:
                - Query projection
                - Key projection
                - Value projection
                - Output projection
            attn (Tensor): To store the attention weights for visualization or further processing.
            dropout (nn.Dropout): Dropout layer.
    Note:
        Use the above given `clones` function to create multiple instances of a module.
        Use the `attention` function to compute the attention scores and apply the softmax.
        Do not use the `nn.MultiheadAttention` module for this implementation.
    """

    def __init__(self, h, d_model, dropout=0.1):
        """
        Initializes the MultiHeadedAttention module.
        Args:
            h (int): Number of attention heads.
            d_model (int): Total dimension of the model.
            dropout (float): Dropout rate.
        """
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0, "d_model must be divisible by h"
        # TODO: Define multi-headed attention class initialization
        # YOUR CODE STARTS HERE

        self.d_k = d_model // h  # Dimension of each key/query/value vector per head
        self.h = h  # Number of heads
        self.d_model = d_model  # Total dimension of the model
        self.linear_layers = clones(nn.Linear(d_model, d_model), 4)  # 4 linear layers for Q, K, V, and output
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)  # Dropout layer for regularization

        # YOUR CODE ENDS HERE

    def forward(self, query, key, value, mask=None):
        """
        Forward pass of the MultiHeadedAttention module.
        Parameters:
            query (Tensor): Query tensor of shape [batch_size, seq_len, d_model].
            key (Tensor): Key tensor of shape [batch_size, seq_len, d_model].
            value (Tensor): Value tensor of shape [batch_size, seq_len, d_model].
            mask (Tensor, optional): Mask tensor of shape [batch_size, 1, seq_len, seq_len].
        Returns:
            Tensor: The output tensor after applying multi-headed attention and linear transformation.
                    Shape [batch_size, seq_len, d_model].

        Steps to implement:
        1. Perform linear projections in batch from d_model => h x d_k. Shape of each output tensor should be
        [batch_size, seq_len, h, d_k].
        HINT: Use the `view` method to reshape the tensors, and the `transpose` method to swap the dimensions. Don't
        forget to apply the linear layers before.
        2. Apply attention on all projected vectors in batch. Assign the attention weights to `self.attn`. Use the
        `attention` function that you implemented earlier. self.attn will be useful in analysis question.
        3. Concatenate using a view and apply a final linear layer (last of the module list). Shape of the output tensor
        should be [batch_size, seq_len, h * d_k].
        """
        out = None
        # TODO: Implement multi-headed attention forward pass
        # YOUR CODE STARTS HERE

        batch_size = query.size(0)
        query_len = query.size(1)
        key_len = key.size(1)
        value_len = value.size(1)

        # Linear projections
        query = self.linear_layers[0](query).view(batch_size, query_len, self.h, self.d_k).transpose(1, 2)
        key = self.linear_layers[1](key).view(batch_size, key_len, self.h, self.d_k).transpose(1, 2)
        value = self.linear_layers[2](value).view(batch_size, value_len, self.h, self.d_k).transpose(1, 2)


        # Apply attention
        out, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        # Concatenate heads and apply final linear layer
        out = out.transpose(1, 2).contiguous().view(batch_size, query_len, self.h * self.d_k)
        out = self.linear_layers[3](out)

        # YOUR CODE ENDS HERE
        return out

    def set_weights(self, weights=None, biases=None):
        """
        Set the weights and biases of all the linear layers from external tensors.
        Parameters:
            weights (list of Tensors): A list of tensors for the weights of the linear layers.
            biases (list of Tensors): A list of tensors for the biases of the linear layers.
        """
        # TODO: Implement setting weights and biases for all layers
        # YOUR CODE STARTS HERE

        self.linear_layers[0].weight.data = weights[0]
        self.linear_layers[0].bias.data = biases[0]
        self.linear_layers[1].weight.data = weights[1]
        self.linear_layers[1].bias.data = biases[1]
        self.linear_layers[2].weight.data = weights[2]
        self.linear_layers[2].bias.data = biases[2]
        self.linear_layers[3].weight.data = weights[3]
        self.linear_layers[3].bias.data = biases[3]

        # YOUR CODE ENDS HERE


class FeedForward(nn.Module):
    """
    Implements a position-wise feedforward network which is used as a component in
    transformer. This module applies two linear transformations
    with a ReLU activation in between, along with dropout for regularization.
    Attributes:
            w_1 (nn.Linear): The first linear transformation layer.
            w_2 (nn.Linear): The second linear transformation layer that projects the output
                             back to the model's dimension.
            dropout (nn.Dropout): Dropout layer applied to the activations.
    """

    def __init__(self, d_model, d_ff, dropout=0.1):
        """
        Initializes the FeedForward network with specified dimensions and dropout.
        Args:
            d_model (int): The size of the input and output dimensions.
            d_ff (int): The dimensionality of the hidden layer.
            dropout (float): Dropout rate.
        """
        super(FeedForward, self).__init__()
        # TODO: Define the feedforward class initialization
        # YOUR CODE STARTS HERE

        self.w_1 = nn.Linear(d_model, d_ff)  # First linear layer
        self.w_2 = nn.Linear(d_ff, d_model)  # Second linear layer
        self.dropout = nn.Dropout(p=dropout)  # Dropout layer for regularization
        self.relu = nn.ReLU()  # ReLU activation function

        # YOUR CODE ENDS HERE

    def forward(self, x):
        """
        Defines the computation performed at every call of the feedforward network.
        Parameters:
            x (torch.Tensor): The input tensor with shape [batch_size, sequence_length, d_model].

        Returns:
            torch.Tensor: The output tensor with the same shape as input tensor, after being
                          processed by two linear layers and dropout.
        Apply dropout after the ReLU activation, which is added between w_1 and w_2.
        """
        out = None
        # TODO: Implement the feedforward forward pass
        # YOUR CODE STARTS HERE

        out = self.w_1(x)  # Apply first linear layer
        out = self.relu(out)  # Apply ReLU activation
        out = self.dropout(out)  # Apply dropout
        out = self.w_2(out)  # Apply second linear layer

        # YOUR CODE ENDS HERE
        return out
    
    def set_weights(self, weights=None, biases=None):
        """
        Set the weights and biases of the linear layers from external tensors.
        Parameters:
            weights (list of Tensors): A list of tensors for the weights of the linear layers (0th index for first layer
            and 1st for second layer).
            biases (list of Tensors): A list of tensors for the biases of the linear layers. (0th index for first layer
            and 1st for second layer).
        """
        # TODO: Implement setting weights and biases for all layers
        # YOUR CODE STARTS HERE

        self.w_1.weight.data = weights[0]
        self.w_1.bias.data = biases[0]
        self.w_2.weight.data = weights[1]
        self.w_2.bias.data = biases[1]

        # YOUR CODE ENDS HERE


class LayerNorm(nn.Module):
    """
    Implements a Layer Normalization module as described in the cited literature.
    Attributes:
        scale_param (nn.Parameter): Scale parameter, learnable, initialized to ones.
        shift_param (nn.Parameter): Shift parameter, learnable, initialized to zeros.
        eps (float): A small constant added to the denominator for numerical stability.
    """
    def __init__(self, features, eps=1e-6):
        """
        Initializes the LayerNorm module with the specified number of features and a small
        epsilon value for numerical stability.
        Args:
            features (int): The number of individual features expected in the input.
            eps (float): A small constant to prevent any division by zero during normalization.
        """
        super(LayerNorm, self).__init__()
        # TODO: Define the layer normalization class initialization
        # YOUR CODE STARTS HERE

        self.scale_param = nn.Parameter(torch.ones(features))  # Scale parameter
        self.shift_param = nn.Parameter(torch.zeros(features)) # Shift parameter
        self.eps = eps  # Small constant for numerical stability

        # YOUR CODE ENDS HERE

    def forward(self, x):
        """
        Forward pass of the LayerNorm module.
        Parameters:
            x (torch.Tensor): Input tensor of shape [..., features].
        Returns:
            torch.Tensor: Normalized tensor with the same shape as the input.
        """
        out = None
        # TODO: Implement the layer normalization forward pass
        # YOUR CODE STARTS HERE

        out = (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + self.eps)  # Normalize
        out = self.scale_param * out + self.shift_param  # Scale and shift

        # YOUR CODE ENDS HERE
        return out


class ResidualStreamBlock(nn.Module):
    """
    Implements a residual connection around any sublayer with Layer Normalization.
    Notably, for simplification in this implementation, normalization is applied before
    the sublayer, contrary to some conventional designs where normalization might come
    after the sublayer.
    Attributes:
        norm (LayerNorm): A layer normalization module that normalizes the input.
        dropout (nn.Dropout): A dropout module that randomly zeroes some of the elements
                              of the input tensor with probability `dropout` during training,
                              which helps prevent overfitting.
    """

    def __init__(self, size, dropout):
        """
        Initializes the ResidualStreamBlock module with a specific size for normalization
        and a specified dropout rate.
        Args:
            size (int): The number of features in the input tensors expected by the layer normalization.
            dropout (float): The dropout probability to be used in the dropout layer.
        """
        super(ResidualStreamBlock, self).__init__()
        # TODO: Define the residual stream block class initialization
        # YOUR CODE STARTS HERE

        self.norm = LayerNorm(size)  # Layer normalization for the input
        self.dropout = nn.Dropout(p=dropout)  # Dropout layer for regularization
        self.size = size  # Size of the input features, used for normalization

        # YOUR CODE ENDS HERE

    def forward(self, x, sublayer):
        """
        Forward pass through the ResidualStreamBlock module which applies a residual connection
        followed by a dropout to the output of any sublayer function.
        This is for ease of use because this pattern is common in the transformer architecture.
        Apply dropout after the sublayer has been applied to the normalized input tensor, before skip connection.
        Parameters:
            x (torch.Tensor): The input tensor.
            sublayer (callable): A function or module that processes the normalized input tensor.
        Returns:
            torch.Tensor: The output tensor which is the element-wise addition of the input tensor
                          and the processed output from the sublayer, after dropout has been applied.
        """
        out = None
        # TODO: Implement the residual stream block forward pass
        # YOUR CODE STARTS HERE

        # Normalize the input
        normalized_input = self.norm(x)

        # Apply the sublayer to the normalized input
        sublayer_output = sublayer(normalized_input)

        # Apply dropout to the sublayer output
        sublayer_output = self.dropout(sublayer_output)

        # Add the original input to the sublayer output (residual connection)
        out = x + sublayer_output

        # YOUR CODE ENDS HERE
        return out


class EncoderBlock(nn.Module):
    """
    Represents a single layer of a transformer encoder module. Each encoder layer is composed
    of two sublayers: a self-attention mechanism and a position-wise feed-forward network.
    Each sublayer is wrapped with a residual connection followed by layer normalization.
    Attributes:
        self_attn (nn.Module): The self-attention mechanism component of the encoder.
        feed_forward (nn.Module): The position-wise feed-forward network.
        residual_stream_block1 (ResidualStreamBlock): A residual stream block that wraps the self-attention mechanism.
        residual_stream_block2 (ResidualStreamBlock): A residual stream block that wraps the feed-forward network.
        size (int): Dimensionality of the model, used to ensure consistency in sizes of the
                    input and output of the layer.
    """

    def __init__(self, size, self_attn, feed_forward, dropout):
        """
        Initializes the EncoderLayer with self-attention and feed-forward network along with
        necessary configurations for residual stream blocks.
        Args:
            size (int): The size of the model (i.e., dimensionality of input and output).
            self_attn (nn.Module): An instance of a self-attention mechanism.
            feed_forward (nn.Module): An instance of a position-wise feed-forward network.
            dropout (float): Dropout rate for sublayers within the encoder.
        """
        super(EncoderBlock, self).__init__()
        # TODO: Define the encoder block class initialization
        # YOUR CODE STARTS HERE

        self.size = size  # Size of the model
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.residual_stream_block1 = ResidualStreamBlock(size, dropout)
        self.residual_stream_block2 = ResidualStreamBlock(size, dropout)

        # YOUR CODE ENDS HERE

    def forward(self, x, mask):
        """
        Processes input through one encoder layer of a transformer model following the
        architecture specified in the original transformer paper.
        Don't forget the mask. It will handle the padding tokens, so they are not attended to.
        Parameters:
            x (torch.Tensor): The input tensor to the encoder layer.
            mask (torch.Tensor): The mask tensor to be applied during self-attention to
                                 prevent attention to certain positions.
        Returns:
            torch.Tensor: The output of the encoder layer after processing through self-attention
                          and feed-forward network with residual connections and normalization.
        """
        out = None
        # TODO: Implement the encoder block forward pass
        # YOUR CODE STARTS HERE

        # Apply self-attention
        out = self.residual_stream_block1(x, lambda x: self.self_attn(x, x, x, mask))

        out = self.residual_stream_block2(out, self.feed_forward)

        # YOUR CODE ENDS HERE
        return out


class Encoder(nn.Module):
    """
    Defines the core encoder which is a stack of N identical encoder blocks.
    Attributes:
        layers (nn.ModuleList): A list of identical layer modules that make up the encoder.
        norm (LayerNorm): A normalization layer applied to the output of the last encoder layer
                          to ensure that the output is normalized before it is passed to the
                          next stage of the model.
    """

    def __init__(self, layer, n_blocks):
        """
        Initializes the Encoder module with a stack of N identical layers and a final
        normalization layer.
        HINTS:
            Use the `clones` function to create N identical layers.
            The LayerNorm is applied to the output of the encoder stack. The layer size should be set accordingly.
            Can you use layer.size?
        Args:
            layer (nn.Module): The layer to be cloned and stacked.
            n_blocks (int): The number of times the layer should be cloned to form the encoder stack.
        """
        super(Encoder, self).__init__()
        # TODO: Define the encoder class initialization
        # YOUR CODE STARTS HERE

        self.layers = clones(layer, n_blocks)  # Create a stack of N identical layers
        self.norm = LayerNorm(layer.size)  # Normalization layer for the output of the encoder
        self.size = layer.size

        # YOUR CODE ENDS HERE

    def forward(self, x, mask):
        """
        Processes the input sequence through each layer in the encoder block sequentially.
        HINT: you can use one loop to go through each layer in the encoder stack.
        Parameters:
            x (torch.Tensor): The input tensor to the encoder.
            mask (torch.Tensor): The mask tensor to be applied during the operations of each layer
                                 to prevent attention to certain positions, typically padding.

        Returns:
            torch.Tensor: The output of the encoder after all layers and the final normalization
                          have been applied.
        """
        out = None
        # TODO: Implement the encoder forward pass
        # YOUR CODE STARTS HERE

        for layer in self.layers:
            x = layer(x, mask)
        out = self.norm(x)  # Apply normalization to the output of the last layer

        # YOUR CODE ENDS HERE
        return out


class DecoderBlock(nn.Module):
    """
    Represents a single layer of a transformer decoder module. Each decoder layer consists of
    three main components: a self-attention mechanism, a cross-attention mechanism where the
    decoder attends to the encoder's output, and a position-wise feed-forward network. Each
    sublayer (self-attention, cross-attention, and feed-forward) is wrapped with a residual
    connection followed by layer normalization.
    Attributes:
            size (int): The size of the input and output dimensions for all sublayers.
            self_attn (nn.Module): The self-attention mechanism within the decoder.
            cross_attn (nn.Module): The cross-attention mechanism where the decoder layers attend to
                                    the encoder's output.
            feed_forward (nn.Module): The feed-forward network.
            residual_stream_block1 (ResidualStreamBlock): A residual stream block that wraps the self-attention mechanism.
            residual_stream_block2 (ResidualStreamBlock): A residual stream block that wraps the cross-attention mechanism.
            residual_stream_block3 (ResidualStreamBlock): A residual stream block that wraps the feed-forward network.
    """

    def __init__(self, size, self_attn, cross_attn, feed_forward, dropout):
        """
        Initializes the DecoderLayer with specified self-attention, cross-attention,
        and feed-forward network along with necessary configurations for residual stream blocks.
        Args:
            size (int): The size of the model (i.e., dimensionality of input and output).
            self_attn (nn.Module): An instance of a self-attention mechanism.
            cross_attn (nn.Module): An instance of a cross-attention mechanism.
            feed_forward (nn.Module): An instance of a position-wise feed-forward network.
            dropout (float): Dropout rate for sublayers within the decoder.
        """
        super(DecoderBlock, self).__init__()
        # TODO: Define the decoder block class initialization
        # YOUR CODE STARTS HERE

        self.size = size  # Size of the model
        self.self_attn = self_attn  # Self-attention mechanism
        self.cross_attn = cross_attn  # Cross-attention mechanism
        self.feed_forward = feed_forward  # Feed-forward network
        self.residual_stream_block1 = ResidualStreamBlock(size, dropout)  # Residual stream block for self-attention
        self.residual_stream_block2 = ResidualStreamBlock(size, dropout)  # Residual stream block for cross-attention
        self.residual_stream_block3 = ResidualStreamBlock(size, dropout)  # Residual stream block for feed-forward


        # YOUR CODE ENDS HERE

    def forward(self, x, memory, src_mask, tgt_mask):
        """
        Processes input through one decoder layer of a transformer model following the architecture
        specified in Figure 1 (right) from the original transformer paper.
        Parameters:
            x (torch.Tensor): The input tensor to the decoder layer.
            memory (torch.Tensor): The output from the last layer of the encoder, serving as memory
                                   for the cross-attention mechanism.
            src_mask (torch.Tensor): The mask tensor for the encoder's output, used during cross-attention.
            tgt_mask (torch.Tensor): The mask tensor for the decoder's input, used during self-attention
                                     to prevent attending to subsequent positions.
        Returns:
            torch.Tensor: The output of the decoder layer after processing through self-attention,
                          cross-attention, and feed-forward network with residual connections and normalization.
        """
        out = None
        # TODO: Implement the decoder block forward pass
        # YOUR CODE STARTS HERE

        # Apply self-attention
        out = self.residual_stream_block1(x, lambda x: self.self_attn(x, x, x, tgt_mask))

        # Apply cross-attention
        out = self.residual_stream_block2(out, lambda x: self.cross_attn(x, memory, memory, src_mask))

        # Apply feed-forward network
        out = self.residual_stream_block3(out, self.feed_forward)

        # YOUR CODE ENDS HERE
        return out


class Decoder(nn.Module):
    """
    Defines a generic N-layer decoder module with masking capabilities for transformer models.
    The decoder is composed of a stack of N identical decoder layers, similar to how encoder layers are stacked in the
    encoder.
    Attributes:
        layers (nn.ModuleList): A list of identical decoder layers. Each layer incorporates
                                self-attention, cross-attention, and a feed-forward network,
                                all wrapped within residual connections and normalization.
        norm (LayerNorm): A normalization layer applied to the output of the last decoder
                          layer to ensure the output is normalized before further processing.
    """

    def __init__(self, layer, n_blocks):
        """
        Initializes the Decoder with N identical layers and a layer normalization step
        at the end.
        Implement it similar to encoder.
        Args:
            layer (nn.Module): The decoder layer to be cloned and stacked.
            n_blocks (int): The number of layers in the decoder stack.
                              or output.
        """
        super(Decoder, self).__init__()
        # TODO: Define the decoder class initialization
        # YOUR CODE STARTS HERE

        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(n_blocks)])
        self.norm = nn.LayerNorm(layer.size)

        # YOUR CODE ENDS HERE

    def forward(self, x, memory, src_mask, tgt_mask):
        """
        Processes the input sequence through each decoder layer in sequence, using the
        output of the encoder as memory.
        Implement it similar to encoder, but remember that the decoder has memory from the encoder.
        Parameters:
            x (torch.Tensor): The input tensor to the decoder.
            memory (torch.Tensor): The output tensor from the encoder which serves as memory
                                   in cross-attention mechanisms.
            src_mask (torch.Tensor): The mask for the encoder output, used in cross-attention.
            tgt_mask (torch.Tensor): The mask for the decoder input, used in self-attention
                                     to prevent attention to subsequent positions.
        Returns:
            torch.Tensor: The output tensor from the decoder after passing through all layers
                          and normalization.
        """
        out = None
        # TODO: Implement the decoder forward pass
        # YOUR CODE STARTS HERE

        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        out = self.norm(x)  # Apply normalization to the output of the last layer

        # YOUR CODE ENDS HERE
        return out


class Generator(nn.Module):
    """
    Implements the final generation step in a sequence-to-sequence model. This module
    typically sits at the end of a decoder to transform the decoder's output into a
    probability distribution over the vocabulary. It does so by applying a linear transformation
    followed by a log softmax operation, making it suitable for subsequent calculation of loss
    during training (e.g., using the negative log-likelihood).
    Attributes:
        linear (nn.Linear): Linear transformation layer that projects the decoder's output
                          to the vocabulary space.
    """

    def __init__(self, d_model, vocab):
        """
        Initializes the Generator module with a linear transformation.
        Args:
            d_model (int): The dimensionality of the input feature space.
            vocab (int): The size of the vocabulary for the output space.
        """
        super(Generator, self).__init__()
        # TODO: Define the generator class initialization
        # YOUR CODE STARTS HERE

        self.linear = nn.Linear(d_model, vocab)  # Linear layer to project to vocabulary size
        self.log_softmax = nn.LogSoftmax(dim=-1)  # Log softmax layer

        # YOUR CODE ENDS HERE

    def forward(self, x):
        """
        Defines the forward pass of the Generator. Applies a linear transformation to the input
        tensor and then performs a log softmax on the result to produce a distribution over the
        vocabulary.
        Parameters:
            x (torch.Tensor): The input tensor containing features from the decoder.
        Returns:
            torch.Tensor: The log probability of each vocabulary token for each sequence in the batch.
        """
        out = None
        # TODO: Implement the generator forward pass
        # YOUR CODE STARTS HERE

        out = self.linear(x)  # Apply linear transformation to project to vocabulary size
        out = self.log_softmax(out)  # Apply log softmax to get log probabilities

        # YOUR CODE ENDS HERE
        return out
    
    def set_weights(self, weight=None, bias=None):
        """
        Set the weights and biases of the linear layer from external tensors.
        Parameters:
            weight (Tensor): A tensor for the weights of the linear layer.
            bias (Tensor): A tensor for the biases of the linear layer.
        """
        # TODO: Implement setting weights and biases for the linear layer
        # YOUR CODE STARTS HERE

        self.linear.weight.data = weight  # Set the weights of the linear layer
        self.linear.bias.data = bias  # Set the biases of the linear layer

        # YOUR CODE ENDS HERE


class Transformer(nn.Module):
    """
    Implements a standard Encoder-Decoder transformer. It
    combines an encoder and a decoder with embedding layers for the source and target
    sequences, and a final generator layer that typically produces probabilities over
    a target vocabulary.
    Attributes:
        encoder (nn.Module): The encoder module which processes the input sequence.
        decoder (nn.Module): The decoder module which generates the output sequence.
        src_embed (nn.Module): An embedding layer for the source sequence.
        tgt_embed (nn.Module): An embedding layer for the target sequence.
        generator (nn.Module): A generator layer that converts the output of the decoder
                               into a probability distribution over the target vocabulary.
    """

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        """
        Initializes the EncoderDecoder model with its constituent components.
        Args:
            encoder (nn.Module): The encoder module.
            decoder (nn.Module): The decoder module.
            src_embed (nn.Module): Embedding layer for the source text.
            tgt_embed (nn.Module): Embedding layer for the target text.
            generator (nn.Module): Output generator layer.
        """
        super(Transformer, self).__init__()
        # TODO: Define the transformer class initialization
        # YOUR CODE STARTS HERE

        self.encoder = encoder  # The encoder module
        self.decoder = decoder  # The decoder module
        self.src_embed = src_embed  # Embedding layer for the source sequence
        self.tgt_embed = tgt_embed  # Embedding layer for the target sequence
        self.generator = generator  # Generator layer to produce output probabilities

        # YOUR CODE ENDS HERE

    def forward(self, src, tgt, src_mask, tgt_mask):
        """
        Defines the forward pass of the Encoder-Decoder model using the provided source
        and target sequences along with their respective masks.
        Parameters:
            src (torch.Tensor): The source sequence input tensor.
            tgt (torch.Tensor): The target sequence input tensor.
            src_mask (torch.Tensor): The mask tensor for the source sequence.
            tgt_mask (torch.Tensor): The mask tensor for the target sequence.
        Returns:
            torch.Tensor: The output from the decoder which is then passed to the generator.
        """
        out = None
        # TODO: Implement the transformer forward pass
        # YOUR CODE STARTS HERE

        # Encode the source sequence
        memory = self.encode(src, src_mask)

        # Decode the target sequence
        out = self.decode(memory, src_mask, tgt, tgt_mask)

        # Pass the output through the generator
        out = self.generator(out)

        # YOUR CODE ENDS HERE
        return out

    def encode(self, src, src_mask):
        """
        Encodes the source sequence.
        Parameters:
            src (torch.Tensor): The source sequence tensor.
            src_mask (torch.Tensor): The mask tensor for the source sequence.
        Returns:
            torch.Tensor: The encoded output, which serves as the context for the decoder.
        """
        out = None
        # TODO: Implement the encoding function
        # YOUR CODE STARTS HERE

        # Embed the source sequence
        src_embedded = self.src_embed(src)

        # Pass through the encoder
        out = self.encoder(src_embedded, src_mask)

        # YOUR CODE ENDS HERE
        return out

    def decode(self, memory, src_mask, tgt, tgt_mask):
        """
        Decodes the target sequence using the encoded source as context.
        Parameters:
            memory (torch.Tensor): The output from the encoder.
            src_mask (torch.Tensor): The mask for the source sequence, used in the decoder.
            tgt (torch.Tensor): The target sequence tensor.
            tgt_mask (torch.Tensor): The mask tensor for the target sequence.
        Returns:
            torch.Tensor: The output from the decoder.
        """
        out = None
        # TODO: Implement the decoding function
        # YOUR CODE STARTS HERE

        # Embed the target sequence
        tgt_embedded = self.tgt_embed(tgt)

        # Pass through the decoder
        out = self.decoder(tgt_embedded, memory, src_mask, tgt_mask)

        # YOUR CODE ENDS HERE
        return out


def make_model(src_vocab, tgt_vocab, n_blocks=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
    """
    Constructs a Transformer model using specified hyperparameters and initializes it.
    Parameters:
        src_vocab (int): Size of the source vocabulary.
        tgt_vocab (int): Size of the target vocabulary.
        n_blocks (int, optional): Number of blocks in both the encoder and decoder. Default is 6.
        d_model (int, optional): Dimensionality of the input embeddings. Default is 512.
        d_ff (int, optional): Dimensionality of the feed-forward layer. Default is 2048.
        h (int, optional): Number of attention heads. Default is 8.
        dropout (float, optional): Dropout rate. Default is 0.1.
    Returns:
        nn.Module: A Transformer model configured with the specified hyperparameters.
    The model construction includes multi-head attention mechanisms, feed-forward networks,
    positional encodings for inputs, and embeddings for both source and target vocabularies.
    All parameters are initialized using the Xavier uniform initialization, which is crucial
    for deep learning models as it helps in maintaining a level of variance that is neither
    too small nor too large.
    """
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = FeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = Transformer(
        Encoder(EncoderBlock(d_model, c(attn), c(ff), dropout), n_blocks),
        Decoder(DecoderBlock(d_model, c(attn), c(attn), c(ff), dropout), n_blocks),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab)
    )

    # Initialize parameters with Xavier uniform (also known as Glorot initialization).
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return model
