import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class LightGCN(nn.Module):
    def __init__(self, num_users, num_items, norm_adj, embed_dim=64, n_layers=3, dropout=0.1):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.n_layers = n_layers
        self.embed_dim = embed_dim
        self.dropout = dropout
        
        # Initialize embeddings
        self.embedding_user = nn.Embedding(num_users, embed_dim)
        self.embedding_item = nn.Embedding(num_items, embed_dim)
        
        # Xavier initialization
        nn.init.xavier_uniform_(self.embedding_user.weight)
        nn.init.xavier_uniform_(self.embedding_item.weight)
        
        # Convert adjacency matrix to tensor
        self.adj = self._convert_sp_mat_to_tensor(norm_adj)
        
        # Dropout layer
        self.dropout_layer = nn.Dropout(dropout)
    
    def to(self, device):
        """Move model to device and also move adjacency matrix"""
        super().to(device)
        if hasattr(self, 'adj'):
            self.adj = self.adj.to(device)
        return self

    def _convert_sp_mat_to_tensor(self, mat):
        coo = mat.tocoo()
        # Convert to numpy array first to avoid the warning
        indices = np.vstack([coo.row, coo.col])
        indices = torch.LongTensor(indices)
        values = torch.FloatTensor(coo.data)
        return torch.sparse_coo_tensor(indices, values, coo.shape)

    def forward(self):
        # Initial embeddings
        x = torch.cat([self.embedding_user.weight, self.embedding_item.weight], dim=0)
        all_emb = [x]
        
        # LightGCN propagation
        for _ in range(self.n_layers):
            x = torch.sparse.mm(self.adj, x)
            x = self.dropout_layer(x)
            all_emb.append(x)
        
        # Layer combination (mean pooling)
        out = torch.stack(all_emb, dim=0).mean(0)
        user_emb, item_emb = out[:self.num_users], out[self.num_users:]
        
        return user_emb, item_emb

    def predict(self, users, items):
        user_emb, item_emb = self.forward()
        u = user_emb[users]
        i = item_emb[items]
        return (u * i).sum(dim=1)
    
    def get_embeddings(self):
        """Get user and item embeddings"""
        return self.forward()
    
    def compute_loss(self, users, pos_items, neg_items):
        """Compute BPR loss"""
        user_emb, item_emb = self.forward()
        
        u = user_emb[users]
        pos_i = item_emb[pos_items]
        neg_i = item_emb[neg_items]
        
        pos_scores = (u * pos_i).sum(dim=1)
        neg_scores = (u * neg_i).sum(dim=1)
        
        # BPR loss
        loss = -torch.mean(F.logsigmoid(pos_scores - neg_scores))
        
        # L2 regularization
        l2_reg = torch.norm(u, p=2, dim=1).mean() + torch.norm(pos_i, p=2, dim=1).mean() + torch.norm(neg_i, p=2, dim=1).mean()
        loss += 1e-4 * l2_reg
        
        return loss