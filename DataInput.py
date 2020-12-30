import os.path
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import gensim
from gensim.utils import simple_preprocess
from eunjeon import Mecab
import re
import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

class Configuration:
    # def __init__(self):
    #     self.date = self.make_save_path()
    #     self.data_path = self.date + '/analysis/'
    #     self.model_path = self.date + '/model_doc2vec/'
    #     self.tm_model_path = self.date + '/model_tm/'
    #     self.data_file_name = self.get_file_name()
    #     self.factor = self.get_factor()

    def __init__(self, filename = None, date=None):
        if filename is not None and date is not None:
            self.date = '1229'
            self.data_path = 'analysis/' + self.date + '/data/'
            # self.model_path = self.date + '/model_doc2vec/'
            self.tm_model_path = 'analysis/' + self.date + '/model_tm/'
            self.data_file_name = 'patent_' + filename
        else:
            self.date = '1229'
            self.data_path = 'analysis/' + self.date + '/data/'
            # self.model_path = self.date + '/model_doc2vec/'
            self.tm_model_path = 'analysis/' + self.date + '/model_tm/'
            self.data_file_name = self.get_file_name()
            # self.factor = self.get_factor()


    def get_file_name(self):
        file_name = input(' > file_name : ')
        return 'patent_' + file_name

    def make_save_path(self): ## directory는 'models/날짜'의 형식으로 설정해야 함
        print('==== Configuration ====')
        directory = 'analysis/doc2vec_test_data/' + input('analysis date : ')
        if not os.path.exists(directory):
            os.makedirs(directory)
        return directory

class DataInput:
    def __init__(self, config):
        self.data_path = config.data_path
        self.file_name = config.data_file_name
        self.document_title, self.abstracts = self.pre_prosseccing()


    def make_ngram(self, text, n):  ## n == 3 --> trigram, n==2 --> bigram
        # min_count : Ignore all words and bigrams with total collected count lower than this value.
        # threshold : Represent a score threshold for forming the phrases (higher means fewer phrases).
        #             A phrase of words a followed by b is accepted if the score of the phrase is greater than threshold.
        #             Heavily depends on concrete scoring-function, see the scoring parameter.
        if n == 2:
            print(' ...make bigram...')
            bigram = gensim.models.Phrases(text, min_count=5, threshold=20.0)
            bigram_mod = gensim.models.phrases.Phraser(bigram)
            return [bigram_mod[doc] for doc in text]
        elif n == 3:
            print(' ...make trigram...')
            bigram = gensim.models.Phrases(text, min_count=5, threshold=20.0)
            bigram_mod = gensim.models.phrases.Phraser(bigram)
            trigram = gensim.models.Phrases(bigram[text], threshold=20.0)
            trigram_mod = gensim.models.phrases.Phraser(trigram)
            return [trigram_mod[bigram_mod[doc]] for doc in text]

    def clean_punt(self, text):
        for p in self.mapping:
            text = text.replace(p, self.mapping[p])

        for p in self.punct:
            text = text.replace(p, f' {p} ')

        specials = {'\u200b': ' ', '…': ' ... ', '\ufeff': '', 'करना': '', 'है': ''}

        for s in specials:
            text = text.replace(s, specials[s])

        return text.strip()

    def data_text_cleansing(self, texts):
        print(' ...Run text cleanning...')
        sentence = []
        data = texts
        # # 영문자 이외의 문자는 공백으로 변환
        # analysis = [re.sub('[^a-zA-Z]', ' ', str(sent)) for sent in analysis]
        #
        # for sent in analysis:
        #     print(sent)

        # Remove emails
        data = [re.sub('\S*@\S*\s?', '', str(sent)) for sent in data]

        # Remove new line characters
        data = [re.sub('\s\s+', ' ', str(sent)) for sent in data]

        # Remove distracting single quotes
        data = [re.sub('\'', '', sent) for sent in data]

        data = [sent.replace('\n', ' ') for sent in data]

        return data

    def sent_to_words(self, sentences):
        for sentence in sentences:
            yield (gensim.utils.simple_preprocess(str(sentence), min_len=1, deacc=True))

    def lematization(self, texts): #['NOUN', 'ADJ', 'VERB', 'ADV']
        print(' ...Make lematization...')
        texts_out = []
        mecab = Mecab()
        for sent in tqdm(texts):
            sent = ' '.join(sent)
            sent = mecab.nouns(sent)
            texts_out.append(sent)
        return texts_out

    def pre_prosseccing(self):
        print('==== Preprocessing ====')
        data = pd.read_csv(self.data_path+self.file_name+'.csv', encoding='utf-8')
            # (file=self.data_path + self.data_file_name+'.csv', encoding='utf-8')
        abstract_texts = data['abstract'].tolist()
        with open(self.data_path + 'data_result/'+  self.file_name+'.documents', 'wb') as f:
            pickle.dump(abstract_texts, f)
        sentences = self.data_text_cleansing(abstract_texts)
        data_words = list(self.sent_to_words(sentences))
        morphs_words = self.lematization(data_words)
        bigram = self.make_ngram(morphs_words, n=2)
        for i in range(len(bigram)):
            print(f'[{i}] : {bigram[i]}')

        with open(self.data_path + 'data_result/'+self.file_name+'.corpus', 'wb') as f:
            pickle.dump(bigram, f)

        document_id = data['patentTitle']
        print('=== end preprocessing ===')
        return document_id, bigram