import json
import torch
from torch.utils.data import Dataset

PAD_ID = 0
SOS_ID = 1
EOS_ID = 2
SEP_ID = 32000

class RoboScholarDataset(Dataset):

    def __init__(self, qa_pairs_path, tokenizer, max_len=512):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.data = []
        with open(qa_pairs_path) as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data[index]
        context_ids  = self.tokenizer.encode(row["relevant_excerpt"]).ids
        question_ids = self.tokenizer.encode(row["question"]).ids
        answer_ids   = self.tokenizer.encode(row["answer"]).ids

        input_ids = [SOS_ID] + context_ids + [SEP_ID] + question_ids + [SEP_ID] + answer_ids + [EOS_ID]

        prefix_len = 1 + len(context_ids) + 1 + len(question_ids) + 1
        input_ids = input_ids[:self.max_len]
        prefix_len = min(prefix_len, self.max_len)

        pad_len = self.max_len - len(input_ids)
        input_ids = input_ids + [PAD_ID] * pad_len

        target_ids = [-100] * self.max_len
        ans_start = prefix_len - 1
        ans_end = min(prefix_len + len(answer_ids), self.max_len)
        for i in range(ans_start, ans_end):
            target_ids[i] = input_ids[i + 1]

        mask = [1] * (self.max_len - pad_len) + [0] * pad_len

        return {
            "input_ids":  torch.tensor(input_ids,  dtype=torch.long),
            "target_ids": torch.tensor(target_ids, dtype=torch.long),
            "mask":       torch.tensor(mask,        dtype=torch.long),
        }