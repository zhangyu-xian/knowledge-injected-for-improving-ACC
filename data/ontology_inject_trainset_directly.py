"""
this code inject the ontology model knowledge into
train dataset. add the description of entity

"""

from usingHuggingfaceTrans.ontologyProcess.test_getOntologyInformation \
    import is_class_in_ontology, get_words_same_hierarchy_with_word
from usingHuggingfaceTrans.trainWord2Vect.corpus_process_and_train_model import \
    get_highest_word_in_list
import pandas as pd
import jieba
from owlready2 import *
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


def get_ontInformation_list(onto_path):
    onto = get_ontology(onto_path).load()
    stdo_classes = list(cls.name for cls in onto.classes())  # get the names of entities in ontology
    instances_list = list(instance.name for instance in onto.individuals())
    object_properties_list = list(object_property.name for object_property in onto.object_properties())
    data_properties_list = list(data_property.name for data_property in onto.data_properties())
    return stdo_classes, instances_list, object_properties_list, data_properties_list


def create_ontology_supply_sentence(sentence, onto_path):
    supply_sentence_list = ['<s>' + sentence]
    onto = get_ontology(onto_path).load()
    stdo_classes, instances_list, object_properties_list, data_properties_list = get_ontInformation_list(onto_path)
    seg_words_list = seg_sentence_with_jieba(sentence)
    onto_label_list = []
    for i in range(len(seg_words_list)):
        if seg_words_list[i] in stdo_classes:
            onto_label_list.append('class')
        elif seg_words_list[i] in instances_list:
            onto_label_list.append('instance')
        elif seg_words_list[i] in object_properties_list:
            onto_label_list.append('object_property')
        elif seg_words_list[i] in data_properties_list:
            onto_label_list.append('data_property')
        else:
            onto_label_list.append('None')
    sen_words_onto_inf_dictionary = dict(zip(seg_words_list, onto_label_list))
    for key in sen_words_onto_inf_dictionary:
        if sen_words_onto_inf_dictionary[key] == 'class':
            supply_sentence_list.append('<s>' + key)
        elif sen_words_onto_inf_dictionary[key] == 'instance':
            class_of_instance = onto[key].is_a[0].name
            supply_sentence = key + '是一种' + class_of_instance
            supply_sentence_list.append('<s>' + supply_sentence)
        elif sen_words_onto_inf_dictionary[key] == 'data_property':
            domain_words_has_data_property = []
            domain_words_has_data_property_dis = []
            for word in seg_words_list:
                if (word in stdo_classes and onto[word] in onto[key].domain) or \
                        (word in instances_list and onto[word].is_a[0] in onto[key].domain):
                    domain_words_has_data_property.append(word)
                    domain_words_has_data_property_dis.append(
                        abs(seg_words_list.index(word) - seg_words_list.index(key)))
            if len(domain_words_has_data_property) > 0:
                domain_word_has_data_property = domain_words_has_data_property[
                    domain_words_has_data_property_dis.index(min(domain_words_has_data_property_dis))
                ]
                supply_sentence = domain_word_has_data_property + '具有属性' + key
                supply_sentence_list.append('<s>' + supply_sentence)
            else:
                supply_sentence = key + '是数据属性'
                supply_sentence_list.append('<s>' + supply_sentence)
    return supply_sentence_list


if __name__ == '__main__':
    ontology_path = 'E:/all_project/chuanqin/shieldTunnelDesignChinese1.owl'
    # ontoPath = 'E:/all_project/chuanqin/shieldTunnelDesignChinese1.owl'
    # onto = get_ontology(ontoPath).load()
    # sen_text = '水工盾构隧道的内水压力应呈现单一压力状态。'
    # sen_words_onto_inf_dictionary = create_ontology_supply_sentence(sen_text, ontoPath)
    # print(sen_words_onto_inf_dictionary)
    # a = [5, 2, 8, 1, 4]
    # print(a.index(min(a)))
    train_dataset_file_path = './testDataset.csv'
    x_list, y_list = read_data1(train_dataset_file_path)
    x_list_with_supply_sentence = []
    for i in range(len(x_list)):
        sentence_x = x_list[i]
        supply_sentences_list = create_ontology_supply_sentence(sentence_x, ontology_path)
        supply_sentence_x = ''.join(supply_sentences_list)
        x_list_with_supply_sentence.append(supply_sentence_x)
    output_file_path = './testDataset_with_supply_ontology_inf.csv'
    with open(output_file_path, 'w', encoding="utf_8_sig",
              newline="",
              errors="ignore") as csvfile:  # 
        fieldnames = ['Text', 'Translation']
        write = csv.DictWriter(csvfile, fieldnames=fieldnames)
        write.writeheader()  # 写表头
        for i in range(len(x_list_with_supply_sentence)):
            write.writerow({'Text': x_list_with_supply_sentence[i],
                            'Translation': y_list[i]})
