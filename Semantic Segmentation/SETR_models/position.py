import torch
import torch.nn as nn

class LearnedPosEmbedding(nn.Module):
    def __init__(self, embedding_dim, max_num_embedding, seq_len):
        super(LearnedPosEmbedding, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=max_num_embedding, embedding_dim=embedding_dim)
        self.seq_len = seq_len

        # register a state of the data's positional index in the embedding layer.
        self.register_buffer("Positional_IDs", torch.arange(max_num_embedding).expand((1, -1)))
    
    def forward(self, x, position_id=None):

        #  if there are no positional indices provided, get it from the state
        if position_id is None:  
            position_id = self.Positional_IDs[:, : self.seq_len]
        
        # get the actual position embedding features, and concat on the input
        return x + self.embedding(position_id)
        

class FixedPosEmbedding(nn.Module):
    def __init__(self, embedding_dim, max_num_embedding=5000):
        super(FixedPosEmbedding, self).__init__()

        embedding = torch.zeros(max_num_embedding, embedding_dim)
        position = torch.arange(0, max_num_embedding, dtype=torch.float).unsqueeze(1)

        # using polar coords
        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / embedding_dim)
        )
        embedding[:, 0::2] = torch.sin(position * div_term)
        embedding[:, 1::2] = torch.cos(position * div_term)
        embedding = embedding.unsqueeze(0).transpose(0, 1)

        # register a state of the data's embedding layer.
        self.register_buffer('embedding', embedding)

    def forward(self, x):
        return x + self.embedding[: x.size(0), :]
