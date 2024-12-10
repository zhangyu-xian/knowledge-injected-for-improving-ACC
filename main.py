from transformers import AdamW, get_scheduler
# from transformers import
from config import model, tokenizer, \
            learning_rate, epoch_num, \
            train_data_path, test_data_path, model_save_path
from train import train_loop, test_loop
import torch
from data_loader import getDataloader

if __name__ == "__main__":
    train_dataloader = getDataloader(train_data_path)
    valid_dataloader = getDataloader(test_data_path)
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=epoch_num * len(train_dataloader),
    )
    total_loss = 0.
    best_avg_rouge = 0.
    for t in range(epoch_num):
        print(f"Epoch {t + 1}/{epoch_num}\n-------------------------------")
        total_loss = train_loop(train_dataloader, model, optimizer, lr_scheduler, t + 1, total_loss)
        if t>=15:
            single_trans_recall, single_trans_precision, single_trans_f1, blue_score = test_loop(valid_dataloader,
                                                                                                 model)
            # print(valid_rouge)
            # rouge_avg = valid_rouge['avg']
            # 在进行一定程度训练后再更新保存模型
            if single_trans_f1 > best_avg_rouge:
                best_avg_rouge = single_trans_f1
                print('saving new weights...\n')
                torch.save(model.state_dict(), model_save_path + f'/model_weights.bin')
                # torch.save(model.state_dict(), f'./experiment/epoch_{t + 1}_valid_rouge_{rouge_avg:0.4f}_model_weights.bin')
    print("Done!")
