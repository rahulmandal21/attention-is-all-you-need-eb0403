import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerModel(nn.Module):
    """
    A PyTorch implementation of the Transformer model architecture.
    
    The Transformer model consists of an encoder and a decoder, each composed of a stack of identical layers.
    The encoder maps an input sequence to a sequence of continuous representations, and the decoder generates an output sequence one element at a time.
    The model uses multi-head self-attention and point-wise, fully connected feed-forward networks for both the encoder and decoder.
    """

    def __init__(self, d_model: int, num_heads: int, num_layers: int, input_dim: int, output_dim: int):
        """
        Initializes the Transformer model.
        
        Args:
        d_model (int): The dimensionality of the model.
        num_heads (int): The number of attention heads.
        num_layers (int): The number of layers in the encoder and decoder.
        input_dim (int): The dimensionality of the input data.
        output_dim (int): The dimensionality of the output data.
        """
        super().__init__()
        self.encoder = TransformerEncoder(d_model, num_heads, num_layers, input_dim)
        self.decoder = TransformerDecoder(d_model, num_heads, num_layers, output_dim)

    def forward(self, input_seq: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass through the Transformer model.
        
        Args:
        input_seq (torch.Tensor): The input sequence.
        
        Returns:
        torch.Tensor: The output sequence.
        """
        encoder_output = self.encoder(input_seq)
        decoder_output = self.decoder(encoder_output)
        return decoder_output


class TransformerEncoder(nn.Module):
    """
    A PyTorch implementation of the Transformer encoder.
    
    The Transformer encoder consists of a stack of identical layers, each comprising two sub-layers: multi-head self-attention and position-wise fully connected feed-forward networks.
    """

    def __init__(self, d_model: int, num_heads: int, num_layers: int, input_dim: int):
        """
        Initializes the Transformer encoder.
        
        Args:
        d_model (int): The dimensionality of the model.
        num_heads (int): The number of attention heads.
        num_layers (int): The number of layers in the encoder.
        input_dim (int): The dimensionality of the input data.
        """
        super().__init__()
        self.input_embedding = nn.Linear(input_dim, d_model)
        self.layers = nn.ModuleList([TransformerEncoderLayer(d_model, num_heads) for _ in range(num_layers)])

    def forward(self, input_seq: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass through the Transformer encoder.
        
        Args:
        input_seq (torch.Tensor): The input sequence.
        
        Returns:
        torch.Tensor: The output sequence.
        """
        input_seq = self.input_embedding(input_seq)
        for layer in self.layers:
            input_seq = layer(input_seq)
        return input_seq


class TransformerEncoderLayer(nn.Module):
    """
    A PyTorch implementation of a single layer in the Transformer encoder.
    
    Each layer consists of two sub-layers: multi-head self-attention and position-wise fully connected feed-forward networks.
    """

    def __init__(self, d_model: int, num_heads: int):
        """
        Initializes the Transformer encoder layer.
        
        Args:
        d_model (int): The dimensionality of the model.
        num_heads (int): The number of attention heads.
        """
        super().__init__()
        self.self_attn = nn.MultiHeadAttention(d_model, num_heads)
        self.feed_forward = nn.Linear(d_model, d_model)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)

    def forward(self, input_seq: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass through the Transformer encoder layer.
        
        Args:
        input_seq (torch.Tensor): The input sequence.
        
        Returns:
        torch.Tensor: The output sequence.
        """
        attention_output = self.self_attn(input_seq, input_seq)
        attention_output = self.layer_norm1(input_seq + attention_output[0])
        feed_forward_output = self.feed_forward(attention_output)
        feed_forward_output = self.layer_norm2(attention_output + feed_forward_output)
        return feed_forward_output


class TransformerDecoder(nn.Module):
    """
    A PyTorch implementation of the Transformer decoder.
    
    The Transformer decoder consists of a stack of identical layers, each comprising three sub-layers: multi-head self-attention, encoder-decoder attention, and position-wise fully connected feed-forward networks.
    """

    def __init__(self, d_model: int, num_heads: int, num_layers: int, output_dim: int):
        """
        Initializes the Transformer decoder.
        
        Args:
        d_model (int): The dimensionality of the model.
        num_heads (int): The number of attention heads.
        num_layers (int): The number of layers in the decoder.
        output_dim (int): The dimensionality of the output data.
        """
        super().__init__()
        self.layers = nn.ModuleList([TransformerDecoderLayer(d_model, num_heads) for _ in range(num_layers)])
        self.output_linear = nn.Linear(d_model, output_dim)

    def forward(self, encoder_output: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass through the Transformer decoder.
        
        Args:
        encoder_output (torch.Tensor): The output of the encoder.
        
        Returns:
        torch.Tensor: The output sequence.
        """
        decoder_output = encoder_output
        for layer in self.layers:
            decoder_output = layer(decoder_output, encoder_output)
        decoder_output = self.output_linear(decoder_output)
        return decoder_output


class TransformerDecoderLayer(nn.Module):
    """
    A PyTorch implementation of a single layer in the Transformer decoder.
    
    Each layer consists of three sub-layers: multi-head self-attention, encoder-decoder attention, and position-wise fully connected feed-forward networks.
    """

    def __init__(self, d_model: int, num_heads: int):
        """
        Initializes the Transformer decoder layer.
        
        Args:
        d_model (int): The dimensionality of the model.
        num_heads (int): The number of attention heads.
        """
        super().__init__()
        self.self_attn = nn.MultiHeadAttention(d_model, num_heads)
        self.encoder_attn = nn.MultiHeadAttention(d_model, num_heads)
        self.feed_forward = nn.Linear(d_model, d_model)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.layer_norm3 = nn.LayerNorm(d_model)

    def forward(self, decoder_input: torch.Tensor, encoder_output: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass through the Transformer decoder layer.
        
        Args:
        decoder_input (torch.Tensor): The input to the decoder.
        encoder_output (torch.Tensor): The output of the encoder.
        
        Returns:
        torch.Tensor: The output sequence.
        """
        self_attn_output = self.self_attn(decoder_input, decoder_input)
        self_attn_output = self.layer_norm1(decoder_input + self_attn_output[0])
        encoder_attn_output = self.encoder_attn(self_attn_output, encoder_output)
        encoder_attn_output = self.layer_norm2(self_attn_output + encoder_attn_output[0])
        feed_forward_output = self.feed_forward(encoder_attn_output)
        feed_forward_output = self.layer_norm3(encoder_attn_output + feed_forward_output)
        return feed_forward_output


if __name__ == "__main__":
    model = TransformerModel(d_model=512, num_heads=8, num_layers=6, input_dim=1024, output_dim=1024)
    input_seq = torch.randn(1, 10, 1024)
    output = model(input_seq)
    print(output.shape)