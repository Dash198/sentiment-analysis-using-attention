import torch
import torch.nn as nn

class Attention(nn.Module):
  def forward(self, enc_states, dec_state):
    raise NotImplementedError()

class AdditiveAttention(Attention):
    def __init__(self, enc_dim, dec_dim, att_dim):
        super().__init__()
        self.dec_proj = nn.Linear(dec_dim, att_dim)
        self.enc_proj = nn.Linear(enc_dim, att_dim)
        self.att_proj = nn.Linear(att_dim, 1)

    def forward(self, dec_state, enc_states):
        enc_proj = self.enc_proj(enc_states)
        dec_proj = self.dec_proj(dec_state).unsqueeze(0)
        scores   = self.att_proj(torch.tanh(enc_proj + dec_proj))
        alpha    = torch.softmax(scores.squeeze(2), dim=0)
        ctx      = torch.sum(alpha.unsqueeze(2) * enc_states, dim=0)
        return ctx, alpha

class DotAttention(Attention):
  def __init__(self, enc_dim, dec_dim, att_dim=None):
      super().__init__()
      if enc_dim != dec_dim:
        self.proj = nn.Linear(enc_dim, dec_dim, bias=False)
      else:
        self.proj = None

  def forward(self, dec_state, enc_states):
    if self.proj is not None:
      enc_states = self.proj(enc_states)

    scores = torch.sum(enc_states * dec_state.unsqueeze(0), dim=2)
    alpha = torch.softmax(scores, dim=0)
    ctx = torch.sum(alpha.unsqueeze(2) * enc_states, dim=0)
    return ctx, alpha

class GeneralAttention(Attention):
  def __init__(self, enc_dim, dec_dim, att_dim=None) -> None:
      super().__init__()
      self.W = nn.Linear(enc_dim, dec_dim)

  def forward(self, dec_state, enc_states):
    scores = torch.sum(self.W(enc_states) * dec_state, dim=2)
    alpha = torch.softmax(scores, dim=0)
    ctx = torch.sum(alpha.unsqueeze(2)*enc_states, dim=0)
    return ctx, alpha

class ConcatAttention(Attention):
    def __init__(self, enc_dim, dec_dim, att_dim):
        super().__init__()
        self.Wa = nn.Linear(enc_dim + dec_dim, att_dim)
        self.va = nn.Linear(att_dim, 1)

    def forward(self, dec_state, enc_states):
        seq_len, batch_size, _ = enc_states.shape

        dec_repeat = dec_state.unsqueeze(0).repeat(seq_len, 1, 1)
        concat = torch.cat([enc_states, dec_repeat], dim=2)
        energy = torch.tanh(self.Wa(concat))
        scores = self.va(energy).squeeze(2)

        alpha = torch.softmax(scores, dim=0)
        ctx = torch.sum(alpha.unsqueeze(2) * enc_states, dim=0)

        return ctx, alpha