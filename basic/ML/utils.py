import re
import jieba
from tensorflow.keras.preprocessing.text import Tokenizer


def token(string):
    '''返回所有的文字和数字'''
    return ''.join(re.findall(r'[\d|\w]+', string))

def cut(string):
    '''分词'''
    return ' '.join(jieba.cut(string))

def tokenizer(num_words, strings):
    '''
    构建词语索引和句子编码
    '''
    token = Tokenizer(num_words=num_words)
    # 构建索引
    token.fit_on_texts(strings)
    sequence = token.texts_to_sequences(strings)
    word_index = token.word_index

    return word_index, sequence, token.num_words
