import torch
from torch.utils.data import DataLoader
from config import tokenizer, model, \
    max_input_length, max_target_length, batch_size, device
from tqdm.auto import tqdm
import numpy as np
from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu

rouge = Rouge()
import re


def splitPredicate(text):
    textList = []
    # 按照,划分谓词->,前为)后为中文字符
    # 首先去掉字符串中的空格
    text = text.replace(";(", ",")
    text = text.replace("((", "")
    text = text.replace("))", ")")
    text = text.replace(")))", ")")
    text = text.replace(" ", "")  # 替换掉空格
    text = text.replace(".", "")  # 替换掉句子末端的.
    re_exp = '\),[\u4e00-\u9fa5]'
    isTrueMatch = re.findall(re_exp, text)
    predicateString = ""
    if isTrueMatch:
        predicateText_index = re.finditer(re_exp, text)
        lenIter = sum(1 for _ in re.finditer(re_exp, text))
        index = 0
        iter = 0
        for i in predicateText_index:
            iter += 1
            if index == 0:
                predicateText = text[index:i.span()[0] + 1]
            else:
                predicateText = text[index + 1:i.span()[0] + 1]
            index = i.span()[0] + 1
            textList.append(predicateText)
            if iter == lenIter:
                textList.append(text[i.span()[0] + 2:])
        for i in range(len(textList)):
            if i != len(textList) - 1:
                predicateString = predicateString + textList[i] + " "
            else:
                predicateString = predicateString + textList[i]
    else:
        textList.append(text)
        predicateString = predicateString + text
    return textList, predicateString


def get_index(pre_sen_list, label_sen_list):
    # 输入的是列表
    assert len(pre_sen_list) == len(label_sen_list)
    number_pre = len(pre_sen_list)
    single_trans_recall = single_trans_precision = single_trans_f1 = blue_score = 0
    for i in range(number_pre):
        label_list, label_string = splitPredicate(label_sen_list[i])
        pre_list, pre_string = splitPredicate(pre_sen_list[i])
        rouge_score = rouge.get_scores([pre_string], [label_string])[0]['rouge-1']
        single_sen_blue_score = sentence_bleu([label_list], pre_list, weights=(1, 0, 0, 0))
        single_trans_recall = rouge_score['r'] + single_trans_recall
        single_trans_precision = rouge_score['p'] + single_trans_precision
        single_trans_f1 = rouge_score['f'] + single_trans_f1
        blue_score = blue_score + single_sen_blue_score
    single_trans_recall = '%.3g' % (single_trans_recall / number_pre)
    single_trans_precision = '%.3g' % (single_trans_precision / number_pre)
    single_trans_f1 = '%.3g' % (single_trans_f1 / number_pre)
    blue_score = '%.3g' % (blue_score / number_pre)
    return single_trans_recall, single_trans_precision, single_trans_f1, blue_score


def train_loop(dataloader, model, optimizer, lr_scheduler, epoch, total_loss):
    progress_bar = tqdm(range(len(dataloader)))
    progress_bar.set_description(f'loss: {0:>7f}')
    finish_batch_num = (epoch - 1) * len(dataloader)
    model = model.to(device)
    model.train()
    for batch, batch_data in enumerate(dataloader, start=1):
        batch_data = batch_data.to(device)
        outputs = model(**batch_data)
        # outputs = model(batch_data["input_ids"])
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        total_loss += loss.item()
        progress_bar.set_description(f'loss: {total_loss / (finish_batch_num + batch):>7f}')
        progress_bar.update(1)
    return total_loss


# 测试循环
def test_loop(dataloader, model):
    preds, labels = [], []
    model = model.to(device)
    model.eval()
    for batch_data in tqdm(dataloader):
        batch_data = batch_data.to(device)
        with torch.no_grad():
            generated_tokens = model.generate(
                batch_data["input_ids"],
                attention_mask=batch_data["attention_mask"],
                max_length=max_target_length,
                num_beams=4,
                no_repeat_ngram_size=2,
            ).cpu().numpy()
        if isinstance(generated_tokens, tuple):
            generated_tokens = generated_tokens[0]
        label_tokens = batch_data["labels"].cpu().numpy()

        decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        label_tokens = np.where(label_tokens != -100, label_tokens, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(label_tokens, skip_special_tokens=True)

        preds += [' '.join(pred.strip()) for pred in decoded_preds]
        labels += [' '.join(label.strip()) for label in decoded_labels]

    single_trans_recall, single_trans_precision, single_trans_f1, blue_score = get_index(preds, labels)

    # scores = rouge.get_scores(hyps=preds, refs=labels, avg=True)['rouge-1']
    # result = {key: value['f'] * 100 for key, value in scores.items()}
    # result['avg'] = np.mean(list(result.values()))
    print("single_trans_recall:", single_trans_recall,
          "single_trans_precision:", single_trans_precision,
          "single_trans_f1:", single_trans_f1,
          "blue_score:", blue_score)
    return single_trans_recall, single_trans_precision, single_trans_f1, blue_score


if __name__ == '__main__':
    translatedCodeText = '盾构隧道(_盾构隧道),地质断裂带(_地质断裂带),地裂缝(_地裂缝),不穿越(_盾构隧道,_地质断裂带),不穿越(_盾构隧道,_地裂缝).'
    groundTruthCodeText = '盾构隧道(_盾构隧道),地质断裂带(_地质断裂带),地裂缝(_地裂缝),不穿越(_盾构隧道,_地质断裂带),不穿越(_盾构隧道,_地裂缝).'
    single_trans_recall, single_trans_precision, single_trans_f1, blue_score = get_index(
        translatedCodeText, groundTruthCodeText
    )
    print(single_trans_recall)
    print(single_trans_recall)
    print(single_trans_f1)
    print(blue_score)
