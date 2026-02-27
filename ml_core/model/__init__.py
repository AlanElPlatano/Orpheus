"""
Transformer model architecture and components.
"""

from .embeddings import MusicEmbedding, TokenEmbedding, PositionalEncoding, ScaleDegreeEmbedding, get_embedding_layer
from .transformer import MusicTransformer, create_model
from .constraints import GenerationState

__all__ = [
    'MusicEmbedding',
    'TokenEmbedding',
    'PositionalEncoding',
    'ScaleDegreeEmbedding',
    'get_embedding_layer',
    'MusicTransformer',
    'create_model',
    'GenerationState'
]
