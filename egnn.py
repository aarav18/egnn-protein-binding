import torch
import torch.nn as nn


class EGCL(nn.Module):
    def __init__(self, inp_nf, out_nf, hidden_nf, edges_in_d=0, act_fn=nn.SiLU(), residual=True, attention=False, normalize=False, coords_agg="mean", tanh=False):
        super(EGCL, self).__init__()
        inp_edge = inp_nf * 2
        self.residual = residual
        self.attention = attention
        self.normalize = normalize
        self.coords_agg = coords_agg
        self.tanh = tanh
        
        self.epsilon = 1e-8
        edge_coords_nf = 1
        
        self.edge_mlp = nn.Sequential(
            nn.Linear(inp_edge + edge_coords_nf + edges_in_d, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn
        )
        
        self.node_mlp = nn.Sequential(
            nn.Linear(inp_nf + hidden_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, out_nf)
        )
        
        layer = nn.Linear(hidden_nf, 1, bias=False)
        nn.init.xavier_uniform_(layer.weight, gain=0.001)
        
        coord_mlp = [nn.Linear(hidden_nf, hidden_nf), act_fn, layer]
        if self.tanh:
            coord_mlp.append(nn.Tanh())
        self.coord_mlp = nn.Sequential(*coord_mlp)
        
        if self.attention:
            self.att_mlp = nn.Sequential(
                nn.Linear(hidden_nf, 1),
                nn.Sigmoid()
            )
        
    def edge_model(self, src, tgt, rad, edge_attr):
        if edge_attr is None:
            out = torch.cat([src, tgt, rad], dim=1)
        else:
            out = torch.cat([src, tgt, rad, edge_attr], dim=1)
        
        out = self.edge_mlp(out)
        if self.attention:
            att_val = self.att_mlp(out)
            out = out * att_val
            
        return out

    def node_model(self, x, edge_idx, edge_attr, node_attr):
        row, _ = edge_idx
        agg = unsorted_seg_sum(edge_attr, row, num_segs=x.size(0))
        if node_attr is not None:
            agg = torch.cat([x, agg, node_attr], dim=1)
        else:
            agg = torch.cat([x, agg], dim=1)
        
        out = self.node_mlp(agg)
        if self.residual:
            out = x + out
            
        return out, agg

    def coord_model(self, coord, edge_idx, coord_diff, edge_feat):
        row, _ = edge_idx
        trans = coord_diff * self.coord_mlp(edge_feat)
        if self.coords_agg == "sum":
            agg = unsorted_seg_sum(trans, row, num_segs=coord.size(0))
        elif self.coords_agg == "mean":
            agg = unsorted_seg_mean(trans, row, num_segs=coord.size(0))
        else:
            raise Exception(f"Wrong coords_agg parameter {self.coords_agg}. Must be 'sum' or 'mean'.")
        
        coord += agg
        return coord

    def coord_to_radial(self, edge_idx, coord):
        row, col = edge_idx
        coord_diff = coord[row] - coord[col]
        rad = torch.sum(coord_diff**2, 1).unsqueeze(1)
        
        if self.normalize:
            norm = torch.sqrt(rad).detach() + self.epsilon
            coord_diff = coord_diff / norm
        
        return rad, coord_diff

    def forward(self, h, edge_idx, coord, edge_attr=None, node_attr=None):
        row, col = edge_idx
        rad, coord_diff = self.coord_to_radial(edge_idx, coord)
        
        edge_feat = self.edge_model(h[row], h[col], rad, edge_attr)
        coord = self.coord_model(coord, edge_idx, coord_diff, edge_feat)
        h, _ = self.node_model(h, edge_idx, edge_feat, node_attr)
        
        return h, coord, edge_attr

class EGNN(nn.Module):
    def __init__(self, in_node_nf, hidden_nf, out_node_nf, in_edge_nf=0, device="cpu", act_fn=nn.SiLU(), n_layers=4, residual=True, attention=False, normalize=False, tanh=False):
        super(EGNN, self).__init__()
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        
        self.embedding_in = nn.Linear(in_node_nf, self.hidden_nf)
        self.embedding_out = nn.Linear(self.hidden_nf, out_node_nf)
        
        for i in range(n_layers):
            self.add_module(f"gcl_{i}", EGCL(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_d=in_edge_nf, act_fn=act_fn, residual=residual, attention=attention, normalize=normalize, tanh=tanh))

        self.to(self.device)
    
    def forward(self, h, x, edges, edge_attr):
        h = self.embedding_in(h)
        for i in range(self.n_layers):
            h, x, _ = self._modules[f"gcl_{i}"](h, edges, x, edge_attr=edge_attr)
        h = self.embedding_out(h)
        
        return h, x
            

# helper functions
def unsorted_seg_sum(data, seg_ids, num_segs):
    res_shape = (num_segs, data.size(1))
    res = data.new_full(res_shape, 0)
    seg_ids = seg_ids.unsqueeze(-1).expand(-1, data.size(1))
    res.scatter_add_(0, seg_ids, data)
    return res

def unsorted_seg_mean(data, seg_ids, num_segs):
    res_shape = (num_segs, data.size(1))
    seg_ids = seg_ids.unsqueeze(-1).expand(-1, data.size(1))
    res = data.new_full(res_shape, 0)
    cnt = data.new_full(res_shape, 0)
    res.scatter_add_(0, seg_ids, data)
    cnt.scatter_add_(0, seg_ids, torch.ones_like(data))
    return res / cnt.clamp(min=1)

def get_edges(n_nodes):
    rows, cols = [], []
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j:
                rows.append(i)
                cols.append(j)
    
    edges = [rows, cols]
    return edges

def get_edges_batch(n_nodes, batch_size):
    edges = get_edges(n_nodes)
    edge_attr = torch.ones(len(edges[0]) * batch_size, 1)
    edges = [torch.LongTensor(edges[0]), torch.LongTensor(edges[1])]
    
    if batch_size == 1:
        return edges, edge_attr
    elif batch_size > 1:
        rows, cols = [], []
        for i in range(batch_size):
            rows.append(edges[0] + n_nodes * i)
            cols.append(edges[1] + n_nodes * i)
        edges = [torch.cat(rows), torch.cat(cols)]
    return edges, edge_attr

if __name__ == "__main__":
    # dummy params
    batch_size = 8
    n_nodes = 4
    n_feat = 1
    x_dim = 3
    
    # dummy vars h, x, fc edges
    h = torch.ones(batch_size * n_nodes, n_feat)
    x = torch.ones(batch_size * n_nodes, x_dim)
    edges, edge_attr = get_edges_batch(n_nodes, batch_size)
    
    egnn = EGNN(in_node_nf=n_feat, hidden_nf=32, out_node_nf=1, in_edge_nf=1)
    
    h, x = EGNN(h, x, edges, edge_attr)