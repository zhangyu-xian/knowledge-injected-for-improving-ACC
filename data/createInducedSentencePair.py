from usingHuggingfaceTrans.ontologyProcess.test_getOntologyInformation \
    import is_class_in_ontology, get_words_same_hierarchy_with_word
from usingHuggingfaceTrans.trainWord2Vect.corpus_process_and_train_model import \
    get_highest_word_in_list
import pandas as pd
import jieba
import re
import csv

def read_data1(datapath):
    data = pd.read_csv(datapath, encoding="utf-8")
    x_list = data['Text']
    y_list = data['Translation']
    return x_list, y_list


def seg_sentence_with_jieba(inputSentence):
    '''segment the input sentence with jieba'''
    # add the stopping words
    stop_words_file_path = 'E:/pythonProject/transByML/usingHuggingfaceTrans/trainWord2Vect/stop_words.txt'
    stopwords = {}.fromkeys([line.rstrip() for line in open(stop_words_file_path, encoding='utf-8')])
    jieba.re_han_default = \
        re.compile(r"([\u4E00-\u9FD5a-zA-Z0-9+#&\.\《\》\（\）\/\ \—\-'_%]+)", re.U)
    # load the words
    jieba.load_userdict('E:/pythonProject/transByML/usingHuggingfaceTrans/trainWord2Vect/jieba_word_dictionary.txt')
    seg_list = jieba.cut(inputSentence)
    return list(seg_list)


# iterate each word in seg_words list; get the high words with given word
def get_induced_word(seg_words_list, onto_path):
    induced_sentence_words_list = []
    induced_word_num = 0
    wv_model_path = 'E:/pythonProject/transByML/usingHuggingfaceTrans/trainWord2Vect/word2vec.model'
    for i in range(len(seg_words_list)):
        judeg_result_string = is_class_in_ontology(seg_words_list[i], onto_path)
        if judeg_result_string != "None":
            induced_words_list = get_words_same_hierarchy_with_word \
                (onto_path, judeg_result_string, seg_words_list[i], seg_words_list)
            induced_word = get_highest_word_in_list(wv_model_path, seg_words_list[i], induced_words_list)
            induced_word_num += 1
            induced_sentence_words_list.append(induced_word)
        else:
            induced_word = seg_words_list[i]
            induced_sentence_words_list.append(induced_word)
    return induced_sentence_words_list, induced_word_num


def replace_multiple(text, replacements):
    # regular expression; replacements is dictionary
    pattern = re.compile("|".join(re.escape(key) for key in replacements.keys()))
    # replace the content
    return pattern.sub(lambda match: replacements[match.group(0)], text)


# get the induced sentence and induced translated sentence
def create_induced_induced_sentence_pair(seg_words_list, induced_sentence_words_list, translated_code_text):
    induced_sentence = ''
    induced_translated_sentence = translated_code_text
    assert len(seg_words_list) == len(induced_sentence_words_list)
    for i in range(len(seg_words_list)):
        if seg_words_list[i] != induced_sentence_words_list[i]:
            induced_sentence = ''.join(induced_sentence_words_list)
            replacement_dict = dict(zip(seg_words_list, induced_sentence_words_list))
            induced_translated_sentence = replace_multiple(induced_translated_sentence, replacement_dict)
            break
    return induced_sentence, induced_translated_sentence


# get all the induced sentence-pair
def create_all_induced_sentence_pair(input_train_dataset_path,
                                     output_train_inducedSentencePair_data_path,
                                     onto_path):
    x_list, y_list = read_data1(input_train_dataset_path)
    x_list = list(x_list)
    y_list = list(y_list)
    assert len(x_list) == len(y_list)
    header = ["inducedCodeText", "inducedTranslatedCodeText"]
    inducedCodeText = []
    inducedTranslatedCodeText = []
    induced_sentence_pair=[]
    for i in range(len(x_list)):
        test_code_text = x_list[i]
        test_translated_text = y_list[i]
        seg_word_list = seg_sentence_with_jieba(test_code_text)
        # print(seg_word_list)
        induced_words_list, induced_word_num = get_induced_word(seg_word_list, onto_path)
        induced_sentence, induced_translated_sentence = \
            create_induced_induced_sentence_pair(seg_word_list, induced_words_list, test_translated_text)
        inducedCodeText.append(induced_sentence)
        inducedTranslatedCodeText.append(induced_translated_sentence)
    for source, target in zip(inducedCodeText, inducedTranslatedCodeText):
        induced_sentence_pair.append({
            "inducedCodeText": source,
            "inducedTranslatedCodeText": target,
        })
    with open(output_train_inducedSentencePair_data_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=header)  # 提前预览列名，当下面代码写入数据时，会将其一一对应。
        writer.writeheader()  # 写入列名
        writer.writerows(induced_sentence_pair)  # 写

if __name__ == '__main__':
    datapath = './trainDataset_test.csv'
    inducedSentencePairPath='./transDataset_inducedSentencePair_test.csv'
    shield_tunnel_onto_path = 'E:/all_project/chuanqin/shieldTunnelDesignChinese1.owl'
    # get the training data
    create_all_induced_sentence_pair(datapath,
                                     inducedSentencePairPath,
                                     shield_tunnel_onto_path)