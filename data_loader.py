from torch.utils.data import Dataset
import torch
from torch.utils.data import DataLoader
from config import tokenizer, model, \
    max_input_length, max_target_length, batch_size, no_repeat_ngram_size
from data.createDataset import getDataset

def my_collote_fn(batch_samples):
    batch_inputs, batch_targets = [], []
    for sample in batch_samples:
        batch_inputs.append(sample['codeText'])
        batch_targets.append(sample['prologText'])
    batch_data = tokenizer(
        batch_inputs,
        padding=True,
        max_length=max_input_length,
        truncation=True,
        return_tensors="pt"
    )
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            batch_targets,
            padding=True,
            max_length=max_target_length,
            truncation=True,
            return_tensors="pt"
        )["input_ids"]
        batch_data['decoder_input_ids'] = model.prepare_decoder_input_ids_from_labels(labels)
        end_token_index = torch.where(labels == tokenizer.eos_token_id)[1]
        for idx, end_idx in enumerate(end_token_index):
            labels[idx][end_idx + 1:] = -100
        batch_data['labels'] = labels
    return batch_data


def getDataloader(dataset_path):
    train_data = getDataset(dataset_path)
    valid_data = getDataset(dataset_path)
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=my_collote_fn)
    # valid_dataloader = DataLoader(valid_data, batch_size=batch_size, shuffle=False, collate_fn=my_collote_fn)
    return train_dataloader
