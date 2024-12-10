# coding:utf-8
import re
import jieba
import logging
import gensim
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from owlready2 import *


# process the original text (such as construction organization design file et al.)
def process_corpus(input_path, output_path):
    file1 = open(input_path, 'r', encoding='utf-8')  # 打开要去掉空行的文件
    file2 = open(output_path, 'w', encoding='utf-8')  # 生成没有空行的文件
    for line in file1.readlines():
        line = line.lstrip()
        if line == '\n':
            line = line.strip('\n')  # delete the blank lines
        if (not str(line).isalnum()) and "。" in line:
            sentence_list = re.split(r'。', line)  # split the period
            for sentence in sentence_list:
                if sentence != "":
                    file2.write(sentence)
        elif not str(line).isalnum():
            file2.write(line)

    file1.close()
    file2.close()


# delete line that only contain the figure and characters
def deleteDigitalLineAndCharacterLine(input_path, output_path):
    file1 = open(input_path, 'r', encoding='utf-8')  # 打开要去掉空行的文件
    file2 = open(output_path, 'w', encoding='utf-8')  # 生成没有空行的文件
    for line in file1.readlines():
        if not str(line.strip()).isalnum():
            file2.write(line)
    file1.close()
    file2.close()


# get the comment of object
def get_comment_list(onto, name_string):
    comment_name_list = []
    object_onto = onto[name_string]
    object_onto_comment = object_onto.comment
    if len(object_onto_comment) > 0:
        comment_name_list = object_onto.comment[0].split('\n')
    return comment_name_list


# write the entity, properties and instance into dictionary
def get_ontology_information(ontology_path):
    onto = get_ontology(ontology_path).load()
    # get the names of entities in ontology
    STOnt_classes = list(cls.name for cls in onto.classes())
    # get the name of instances in ontology
    instances_list = list(instance.name for instance in onto.individuals())
    # get the name of object properties in ontology
    object_properties_list = list(object_property.name for object_property in
                                  onto.object_properties())
    # get the name of data properties in ontology
    data_properties_list = list(
        data_property.name for data_property in onto.data_properties())
    ontology_dictionary = []
    ontology_dictionary.extend(STOnt_classes)
    ontology_dictionary.extend(instances_list)
    ontology_dictionary.extend(object_properties_list)
    ontology_dictionary.extend(data_properties_list)
    for object_name in ontology_dictionary[:-1]:
        comment_name_list = get_comment_list(onto, object_name)
        if len(comment_name_list) > 0:
            ontology_dictionary.extend(comment_name_list)
    return ontology_dictionary


# add list into a text file (add the ontology list into the training text)
# add the ontology list into the dictionary
def add_list2Text_file(text_list, textFilePath):
    # open file
    with open(textFilePath, 'a', encoding='utf-8') as f:
        # write elements of list
        for items in text_list:
            f.write('%s\n' % items)
        print("File written successfully")

    # close the file
    f.close()


def seg_with_jieba(infile, outfile):
    '''segment the input file with jieba'''
    with open(infile, 'r', encoding='utf-8') as fin, \
            open(outfile, 'w', encoding='utf-8') as fout:
        # add the stopping words
        stopwords = {}.fromkeys([line.rstrip() for line in open(r'./stop_words.txt', encoding='utf-8')])
        jieba.re_han_default = \
            re.compile(r"([\u4E00-\u9FD5a-zA-Z0-9+#&\.\《\》\（\）\/\ \—\-'_%]+)", re.U)
        # load the words
        jieba.load_userdict('./jieba_word_dictionary.txt')
        for line in fin:
            seg_list = jieba.cut(line)  # 得到分词后的列表
            seg_list_without_stop_words = []
            for seg in seg_list:
                if seg not in stopwords:
                    seg_list_without_stop_words.append(seg)
            seg_res = ' '.join(seg_list_without_stop_words)
            fout.write(seg_res)

    fin.close()
    fout.close()


# training the word2vec model
def word2vec_train(infile, outmodel, vector_size=50, window=5, min_count=3):
    '''train the word vectors by word2vec'''
    # train model
    model = Word2Vec(LineSentence(infile), vector_size=50, window=5, min_count=0, workers=4, sg=1)
    # model = gensim.models.Word2Vec(sentences, size=800, window=5, min_count=3, iter=100, sg=0,
    #                                workers=multiprocessing.cpu_count())
    # model.train(sentences, total_examples=model.corpus_count, epochs = 100) 默认参数，也可以指定
    # save model
    model.save(outmodel)
    # model.wv.save_word2vec_format(outvector, binary=False)


# load trained word2vec model
def load_word2vec_model(w2v_path):
    model = Word2Vec.load(w2v_path)
    return model


# calculate the similar between different words
def calculate_words_similar(model, word1, word2):
    similarity = model.wv.similarity(word1, word2)
    return similarity


# calculate the word vector
def calculate_word_vector(model, word):
    word_embedding = model.wv[word]
    return word_embedding


# get the word that has the highest similarity with the given word
def get_highest_word_in_list(wv_model_path, given_word, word_list):
    model_ = load_word2vec_model(wv_model_path)
    similarity_value = 0.
    highest_similar_word = word_list[0]
    for i in range(len(word_list)):
        similarity_value_words = calculate_words_similar(model_, given_word, word_list[i])
        if similarity_value_words > similarity_value:
            similarity_value = similarity_value_words
            highest_similar_word = word_list[i]
    return highest_similar_word


if __name__ == '__main__':
    # add the ontology list to dictionary file
    ontoPath = 'E:/all_project/chuanqin/shieldTunnelDesignChinese1.owl'
    ontology_list = get_ontology_information(ontoPath)
    dictionaryPath = './jieba_word_dictionary.txt'
    add_list2Text_file(ontology_list, dictionaryPath)

    # # add the ontology into corpus
    # filePath = './result.txt'
    #
    # # segmentation words
    # inputFilePath = './result.txt'
    # outputFilePath = './result_seg.txt'
    # seg_with_jieba(inputFilePath, outputFilePath)
    #
    # # training the word2vec model
    # seg_result_file = './result_seg.txt'
    # model_path = './word2vec.model'
    # word2vec_train(seg_result_file, model_path)

    # load the trained word2vec model
    # model_path = './word2vec.model'
    # model_ = load_word2vec_model('./word2vec.model')
    # word_list=[]
    # for word in model_.wv.index_to_key:
    #     word_list.append(word)
    #     # print(word, model_.wv[word])  # 获得词汇及其对应的向量
    # add_list2Text_file(word_list, './wordsInWord2Vec.txt')
    # print(calculate_words_similar(model_, '盾构隧道', "排气井"))
    # word_list = ['斜螺栓', '市政隧道', '水工隧道', '弹簧', '工作井']
    # given_word = '盾构隧道'
    # highest_similar_word = get_highest_word_in_list(model_path, given_word, word_list)
    # print(highest_similar_word)
