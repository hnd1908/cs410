import torch
from torch.utils.data import Dataset
from pathlib import Path

class BilingualDataset(Dataset):
    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang , seq_len):
        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.seq_len = seq_len

        self.sos_token = torch.tensor([tokenizer_tgt.token_to_id("[SOS]")], dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer_tgt.token_to_id("[EOS]")], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer_tgt.token_to_id("[PAD]")], dtype=torch.int64)

    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, idx):
        item = self.ds[idx]
        src_text = item[self.src_lang]
        tgt_text = item[self.tgt_lang]

        src_tokens = self.tokenizer_src.encode(src_text).ids
        tgt_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        if len(src_tokens) + 2 > self.seq_len or len(tgt_tokens) + 1 > self.seq_len:
            raise ValueError(f"Source or target sequence too long: {len(src_tokens)} (src) + 2, {len(tgt_tokens)} (tgt) + 1 > {self.seq_len}")
        
        # Add start and end tokens
        # [SOS] for target, [EOS] for both
        # [SOS] for source, [EOS] for target
        # [PAD] for padding
        encoder_input = torch.cat([self.sos_token, torch.tensor(src_tokens, dtype=torch.int64), self.eos_token, torch.tensor([self.pad_token] * (self.seq_len - len(src_tokens) - 2), dtype=torch.int64)], dim=0)
        decoder_input = torch.cat([self.sos_token, torch.tensor(tgt_tokens, dtype=torch.int64), torch.tensor([self.pad_token] * (self.seq_len - len(tgt_tokens) - 1), dtype=torch.int64)], dim=0)
        label = torch.cat([torch.tensor(tgt_tokens, dtype=torch.int64), self.eos_token, torch.tensor([self.pad_token] * (self.seq_len - len(tgt_tokens) - 1), dtype=torch.int64)], dim=0)

        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        return {
            "encoder_input": encoder_input,
            "decoder_input": decoder_input,
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(),
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).int() & causal_mask(decoder_input.size(0)),
            "label": label,
            "src_text": src_text,
            "tgt_text": tgt_text,
        }
    
def causal_mask(size):
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    return mask == 0