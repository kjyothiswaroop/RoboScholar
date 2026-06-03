import json
import torch
from pathlib import Path
from torch.utils.data import Dataset
from tokenizers import ByteLevelBPETokenizer

PAD_ID = 0
SOS_ID = 1
EOS_ID = 2
UNK_ID = 3
SEP_ID = 32000

class RoboScholarDataset(Dataset):

    def __init__(self, qa_pairs_path, tokenizer, max_enc_len=256, max_dec_len=128):
        self.tokenizer = tokenizer
        self.max_enc_len = max_enc_len
        self.max_dec_len = max_dec_len
        self.data = []
        with open(qa_pairs_path) as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        row = self.data[index]
        question_ids = self.tokenizer.encode(row["question"]).ids
        context_ids  = self.tokenizer.encode(row["relevant_excerpt"]).ids
        answer_ids   = self.tokenizer.encode(row["answer"]).ids

        encoder_ids = [SOS_ID] + question_ids + [SEP_ID] + context_ids + [EOS_ID]
        decoder_input_ids = [SOS_ID] + answer_ids
        decoder_output_ids = answer_ids + [EOS_ID]

        encoder_ids = encoder_ids[:self.max_enc_len]
        decoder_input = decoder_input_ids[:self.max_dec_len]
        decoder_output = decoder_output_ids[:self.max_dec_len]

        enc_pad_len = self.max_enc_len - len(encoder_ids)
        dec_pad_len = self.max_dec_len - len(decoder_input)

        encoder_ids    = encoder_ids    + [PAD_ID] * enc_pad_len
        decoder_input  = decoder_input  + [PAD_ID] * dec_pad_len
        decoder_output = decoder_output + [PAD_ID] * dec_pad_len

        enc_mask = [1] * (self.max_enc_len - enc_pad_len) + [0] * enc_pad_len
        dec_mask = [1] * (self.max_dec_len - dec_pad_len) + [0] * dec_pad_len

        return {
            "encoder_ids":    torch.tensor(encoder_ids,    dtype=torch.long),
            "enc_mask":       torch.tensor(enc_mask,       dtype=torch.long),
            "decoder_input":  torch.tensor(decoder_input,  dtype=torch.long),
            "decoder_output": torch.tensor(decoder_output, dtype=torch.long),
            "dec_mask":       torch.tensor(dec_mask,       dtype=torch.long),
        }