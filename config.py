import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AdamW, get_scheduler, BertTokenizer

# # greed decode的最大句子长度
# max_len = 60
# # beam size for bleu
# beam_size = 3
# # Label Smoothing
# use_smoothing = False
# # NoamOpt
# use_noamopt = True
tokenizer_list=["bert-base-chinese"]
translation_model_list = ["facebook/mbart-large-50-many-to-many-mmt", "facebook/m2m100_418M"]
text2text_model_list = ["google/mt5-base", "google/mt5-large", "IDEA-CCNL/Randeng-BART-759M-Chinese-BertTokenizer"]
summarization_model_list = ["fnlp/bart-base-chinese", "csebuetnlp/mT5_multilingual_XLSum", "chiakya/Bert-chinese-Summarization"]
textGeneration_model_list = ["mistralai/Mistral-Nemo-Instruct-2407"]

source_data_path = './data/shieldTunnelDesignStandard-WithPrologTranslation2.csv'
train_data_path = './data/trainDataset.csv'
dev_data_path = './data/trainDataset.csv'
test_data_path = './data/testDataset.csv'

tokenizer_checkpoint = translation_model_list[0]
model_checkpoint = translation_model_list[0]
model_name = model_checkpoint.split(r"/")[1]
model_save_path='./experiment/'

tokenizer = AutoTokenizer.from_pretrained(tokenizer_checkpoint, cache_dir=r"e:\transformerModel")
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint, cache_dir=r"e:\transformerModel")
# gpu_id and device id is the relative id
# thus, if you wanna use os.environ['CUDA_VISIBLE_DEVICES'] = '2, 3'
# you should set CUDA_VISIBLE_DEVICES = 2 as main -> gpu_id = '0', device_id = [0, 1]
max_input_length = 512
max_target_length = 512
learning_rate = 3e-5  # 1e-5; 2e-5; 3e-5; 4e-5; 5e-5
epoch_num = 50 # 10 30 50 70
batch_size=4 # 1 2 4 8
device = 'cuda' if torch.cuda.is_available() else 'cpu'
beam_size=2 # 1 2 3 4
no_repeat_ngram_size=0
# gpu_id = '0'
# device_id = [0]
#
# # set device
# if gpu_id != '':
#     device = torch.device(f"cuda:{gpu_id}")
# else:
#     device = torch.device('cpu')

