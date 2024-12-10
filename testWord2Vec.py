# encoding=utf-8
import gensim
from gensim.models import Word2Vec

model = Word2Vec.load('./shieldTunnel_word2vec')
print("模型加载成功！")
print("----词向量维度------")
vec=model.wv['盾构']
print(vec)

