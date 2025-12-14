import torch
from torch import nn
# from dgl.nn import DegreeEncoder

from model.modules import CircuitTranformerLayer, NodeEmbedding


class CircuitTransformer(nn.Module):
    """
    Transformer model for circuit graphs with:
      - Device type used as the basic token.
      - Parameter features (of variable length per node) fused via cross-attention.
      - Laplacian positional encodings added.
    """
    def __init__(
        self,
        num_encoder_layers: int,
        num_decoder_layers: int,
        num_device_types: int,
        num_metrics: int,
        num_cross_layers: int,
        node_emb_dim: int,
        out_dim: int,
        num_heads: int,
        decoder_type: str = "transformer",
        dropout: float = 0.1,
        ffn_embedding_dim: int = 256,
        degree_embed: bool = False,
        max_degree: int = 10,
        activation: str = "gelu",
        norm_type: str = "layernorm",
        max_nodes: int = 100,
        pos_encoding_type: str = "learned",
        lap_dim: int = None,
    ):
        super().__init__()

        self.num_heads = num_heads
        self.node_embed = NodeEmbedding(num_device_types, node_emb_dim)
        # Project scalar parameters to node embedding dim
        self.params_proj = nn.Linear(1, node_emb_dim)
        self.graph_token = nn.Embedding(1, node_emb_dim)

        self.degree_embed = degree_embed
        # if degree_embed:
            # self.degree_encoder = DegreeEncoder(
            #     max_degree=max_degree, embedding_dim=node_emb_dim
            # )

        # Positional embedding
        self.pos_encoding_type = pos_encoding_type
        if pos_encoding_type == "learned":
            self.pos_embed = nn.Embedding(max_nodes, node_emb_dim)
        elif pos_encoding_type == "laplacian":
            if lap_dim is None:
                raise ValueError("For Laplacian positional encoding, lap_dim must be provided.")
            self.lap_proj = nn.Linear(lap_dim, node_emb_dim)
        else:
            raise ValueError(f"Unknown pos_encoding_type: {pos_encoding_type}")

        self.encoder = nn.ModuleList([
            CircuitTranformerLayer(
                d_model=node_emb_dim,
                dim_feedforward=ffn_embedding_dim,
                num_heads=num_heads,
                dropout=dropout,
                activation=activation,
                norm_type=norm_type,
            ) for _ in range(num_encoder_layers - num_cross_layers)] + [
            CircuitTranformerLayer(
                d_model=node_emb_dim,
                dim_feedforward=ffn_embedding_dim,
                num_heads=num_heads,
                dropout=dropout,
                activation=activation,
                norm_type=norm_type,
                cross_attn=True,
            ) for _ in range(num_cross_layers)
        ])

        self.norm = nn.LayerNorm(node_emb_dim)
        self.embed_out = nn.Linear(node_emb_dim, out_dim, bias=False)
        # self.param_embed_out = nn.Linear(node_emb_dim, out_dim, bias=False)

        # Regression decoder for performance metrics.
        self.decoder_type = decoder_type
        if decoder_type == "mlp":
            self.decoder = nn.Sequential(
                nn.Linear(out_dim, out_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(out_dim, num_metrics)
            )
        elif decoder_type == "transformer":
            decoder_layer = nn.TransformerDecoderLayer(d_model=out_dim, nhead=num_heads, dropout=dropout, batch_first=True)
            self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
            # Learnable query embeddings: one per performance metric.
            self.query_embed = nn.Embedding(num_metrics, out_dim)
            # param_layer = nn.TransformerDecoderLayer(d_model=out_dim, nhead=num_heads, dropout=dropout, batch_first=True)
            # self.param_decoder = nn.TransformerDecoder(param_layer, num_layers=1)
            # Final linear layer: for each query, output a scalar prediction.
            self.regressor = nn.Sequential(
                nn.Linear(out_dim, out_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(out_dim, 1)
            )
        else:
            raise ValueError(f"Unknown decoder_type: {decoder_type}")

        # encoder_layer = nn.TransformerEncoderLayer(
        #     d_model=node_emb_dim, nhead=num_heads, dropout=dropout, batch_first=True
        # )
        # # You can experiment with more layers by increasing num_layers.
        # self.param_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        # self.fusion_gate = nn.Linear(node_emb_dim * 2, node_emb_dim)

    def forward(
        self,
        nodes,
        params,
        attn_mask=None,
        in_degree=None,
        out_degree=None,
        lap_pos=None
    ):  
        """
        Args:
            node_feature: [n graph, n node]
            parmas_feature: [n graph, n node,  n params]
            attn_mask: [n graph, n node, n node] Adjacency matrix
        """
        B, N = nodes.shape
        # 1. Embed device tokens.
        node_feature = self.node_embed(nodes)  # (B, N, D)

        # 2. Add positional embedding
        if self.pos_encoding_type == "learned":
            pos_ids = torch.arange(N, device=nodes.device).unsqueeze(0).expand(B, N)
            pos_feature = self.pos_embed(pos_ids)
        elif self.pos_encoding_type == "laplacian":
            if lap_pos is None:
                raise ValueError("lap_pos must be provided when using Laplacian positional encoding.")
            pos_feature = self.lap_proj(lap_pos)
        node_feature = node_feature + pos_feature

        # 3. Optionally add degree-based embeddings.
        if self.degree_embed:
            deg_feature = self.degree_encoder(torch.stack((in_degree, out_degree)))
            node_feature = node_feature + deg_feature

        # 4. Embed parameter features.
        # params: [B, N, P] --> unsqueeze to [B, N, P, 1] then project each scalar.
        B, N, P = params.shape
        params_emb = self.params_proj(params.unsqueeze(-1))  # shape: [B, N, P, node_emb_dim]
        # Flatten the node and parameter dimensions to form a cross-attention sequence.
        params_emb_flat = params_emb.view(B, N * P, -1)  # shape: [B, N*P, node_emb_dim]

        # params_emb_flat_processed = self.param_encoder(params_emb_flat)
        # params_emb_flat = params_emb_flat + params_emb_flat_processed # add residual

        # Create a cross-attention mask for layers using cross attention.
        # We want a mask of shape [B, 1+N, N*P] such that:
        #  - For the graph token (position 0 in x), all parameter tokens are allowed.
        #  - For each node token i (positions 1..N), only its own parameter tokens are allowed.
        cross_attn_mask = torch.full((B, 1 + N, N * P), float('-inf'), device=params.device)
        # Graph token row: allow all parameter tokens.
        cross_attn_mask[:, 0, :] = 0
        # For node tokens: allow only the corresponding block of P tokens.
        for i in range(N):
            cross_attn_mask[:, i + 1, i * P:(i + 1) * P] = 0
        # Expand cross_attn_mask to include num_heads.
        cross_attn_mask = cross_attn_mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
        cross_attn_mask = cross_attn_mask.view(B * self.num_heads, 1 + N, N * P)

        # 5. Prepend a special graph token to the node features.
        graph_token = self.graph_token.weight.unsqueeze(0).expand(B, 1, -1)  # [B, 1, node_emb_dim]
        x = torch.cat([graph_token, node_feature], dim=1)  # Now [B, 1+N, node_emb_dim]

        # 6. Adjust the attention mask.
        if attn_mask is not None:
            graph_mask = torch.ones(B, 1, 1 + N, device=attn_mask.device, dtype=attn_mask.dtype)
            node_mask = torch.cat([torch.ones(B, N, 1, device=attn_mask.device, dtype=attn_mask.dtype), attn_mask], dim=-1)
            attn_mask = torch.cat([graph_mask, node_mask], dim=1)  # Now [B, 1+N, 1+N]
            attn_mask = torch.where(
                attn_mask == 1,
                torch.tensor(0.0, device=attn_mask.device, dtype=attn_mask.dtype),
                torch.tensor(float('-inf'), device=attn_mask.device, dtype=attn_mask.dtype)
            )
            # Expand attn_mask from [B, 1+N, 1+N] to [B*num_heads, 1+N, 1+N]
            attn_mask = attn_mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
            attn_mask = attn_mask.view(B * self.num_heads, 1 + N, 1 + N)

        # 7. Create the key padding mask.
        # The '<PAD>' nodes have a token value of 0.
        key_padding_mask = (nodes == 0)  # shape: [B, N] (True where node is padding)
        key_padding_mask = torch.cat(
            [torch.zeros(B, 1, dtype=torch.bool, device=nodes.device), key_padding_mask],
            dim=1
        ).float()  # shape: [B, 1+N]
        key_padding_mask = torch.where(
            key_padding_mask == 1,
            torch.tensor(float('-inf'), device=key_padding_mask.device),
            torch.tensor(0.0, device=key_padding_mask.device)
        )

        for i, layer in enumerate(self.encoder):
            if i % 2 == 0:  # local attention
                x = layer(
                    x,
                    attn_mask=attn_mask,
                    key_padding_mask=key_padding_mask,
                    cross=params_emb_flat if layer.cross_attn_flag else None,
                    # cross_key_padding_mask=cross_key_padding_mask if layer.cross_attn_flag else None,
                    cross_attn_mask=cross_attn_mask if layer.cross_attn_flag else None,
                    layer = i
                )
            else:   # gloabl attention
                x = layer(
                    x,
                    key_padding_mask=key_padding_mask,
                    cross=params_emb_flat if layer.cross_attn_flag else None,
                    # cross_key_padding_mask=cross_key_padding_mask if layer.cross_attn_flag else None,
                    cross_attn_mask=cross_attn_mask if layer.cross_attn_flag else None,
                    layer = i
                )

        # params_per_node = params_emb_flat.view(B, N, P, -1).mean(dim=2)
        # node_tokens = x[:, 1:]
        # fused_tokens = torch.cat([node_tokens, params_per_node], dim=-1)
        # node_tokens = node_tokens + torch.sigmoid(self.fusion_gate(fused_tokens)) * params_per_node
        # x = torch.cat([x[:, :1], node_tokens], dim=1)

        # param_mem = self.param_embed_out(params_emb_flat)   # [B, N*P, out_dim]
        x = self.embed_out(self.norm(x))
        # memory = torch.cat([x, param_mem], dim=1)

        # graph_pad = torch.zeros(B, 1, dtype=torch.bool, device=nodes.device)
        # node_pad = (nodes == 0)                            # [B, N]
        # param_pad = torch.zeros(B, N * P, dtype=torch.bool, device=nodes.device)
        # mem_pad_mask = torch.cat([graph_pad, node_pad, param_pad], dim=1)

        if self.decoder_type == "mlp":
            perf_metrics = self.decoder(x[:, 0])
            # pooled_embedding = torch.mean(x, dim=1)  # [B, node_emb_dim]
            # perf_metrics = self.decoder(pooled_embedding)
        elif self.decoder_type == "transformer":   
            # Prepare query embeddings for the transformer decoder.
            query_embed = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)
            decoder_out = self.decoder(tgt=query_embed, memory=x, memory_key_padding_mask=key_padding_mask)
            perf_metrics = self.regressor(decoder_out).squeeze(-1) # [B, num_metrics]
            # decoder_out = self.decoder(tgt=query_embed, memory=x, memory_key_padding_mask=key_padding_mask)
            # param_out = self.param_decoder(tgt=decoder_out, memory=param_mem, memory_key_padding_mask=param_pad)
            # perf_metrics = self.regressor(param_out).squeeze(-1) # [B, num_metrics]
        else:
            raise ValueError(f"Unknown decoder_type: {self.decoder_type}")

        return perf_metrics

    def save_checkpoint(self, optimizer, epoch, config, filename):
        """
        Save the checkpoint from within the model.

        Args:
            optimizer (torch.optim.Optimizer): The optimizer instance.
            epoch (int): Current epoch number.
            config (dict): Configuration dictionary.
            filename (str): The file path to save the checkpoint.
        """
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
        }
        torch.save(checkpoint, filename)
        print(f"Checkpoint saved to {filename}")

    def load_checkpoint(self, checkpoint_path, filter_func=None):
        """
        Loads a checkpoint and only updates parameters based on an optional filter function.
        
        Args:
            checkpoint_path (str): Path to the checkpoint file.
            filter_func (callable, optional): A function that takes a parameter key (str) and returns
                True if this parameter should be loaded. If None, all matching parameters are loaded.
                
        Returns:
            dict: The configuration saved in the checkpoint (if any).
        """
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        saved_state_dict = checkpoint.get("model_state_dict", checkpoint)
        current_state_dict = self.state_dict()
        
        # Optionally filter parameters based on the provided filter function.
        if filter_func is not None:
            filtered_state_dict = {k: v for k, v in saved_state_dict.items()
                                   if k in current_state_dict and filter_func(k)}
        else:
            # Default: only load parameters that exist in the current model.
            filtered_state_dict = {k: v for k, v in saved_state_dict.items() if k in current_state_dict}
        
        # Update the current state dict.
        current_state_dict.update(filtered_state_dict)
        # Load the state dict with strict=False to allow missing keys.
        self.load_state_dict(current_state_dict, strict=False)
        
        print("Loaded checkpoint with filtered parameters.")
        # Return saved config if present.
        return checkpoint.get("config", {})



        