from collections import defaultdict

import pyLDAvis
from gensim import corpora
from gensim.models import ldamulticore, CoherenceModel, LdaModel, TfidfModel
import operator
from matplotlib import pyplot as plt
import numpy as np
import pyLDAvis.gensim as gensimvis
import pandas as pd
import pickle

class LDABuilder:
    def __init__(self, config):
        self.data_path = config.data_path
        self.model_path = config.tm_model_path
        self.data_name = config.data_file_name
        self.corpus = self.get_corpus()
        self.documents = self.get_documents()
        self.num_topics = 0
        # self.tag_doc = self.get_tag_doc()
        self.corpus_tfidf, self.dictionary = self.get_corpus_tfidf()

    def get_corpus(self):
        with open(self.data_path + 'data_result/' + self.data_name+'.corpus', 'rb') as f:
            corpus = pickle.load(f)
        documents = []
        adjusted_corpus = []
        for document in corpus:
            tokens = list(set(document))
            documents.append(document)
            adjusted_corpus.append(tokens)

        return adjusted_corpus

    def get_documents(self):
        with open(self.data_path + 'data_result/' + self.data_name+'.documents', 'rb') as f:
            documents = pickle.load(f)
        return documents

    def get_corpus_tfidf(self):
        dictionary = corpora.Dictionary(self.corpus)
        corpus = [dictionary.doc2bow(text) for text in self.corpus]
        tfidf = TfidfModel(corpus)
        corpus_tfidf = tfidf[corpus]
        with open(self.data_path+'data_result/'+self.data_name+'.corpus_tfidf', 'wb') as f:
            pickle.dump(corpus_tfidf, f)
            # pickle.dump(corpus, f)
        # dictionary = corpora.Dictionary(corpus)
        return corpus_tfidf, dictionary
        # return corpus, dictionary

    def getOptimalTopicNum(self):
        # dictionary = corpora.Dictionary(self.corpus)
        # corpus = [dictionary.doc2bow(text) for text in self.corpus]
        # tfidf = TfidfModel(corpus)
        # self.corpus_tfidf = tfidf[corpus]
        # self.dictionary = corpora.Dictionary(self.corpus_tfidf)
        com_nums = []
        for i in range(10, 60, 10):
            if i == 0:
                p = 1
            else:
                p = i
            com_nums.append(p)

        coherence_list = []

        for i in com_nums:
            # lda = gensim.models.ldamodel.LdaModel(corpus=corpus,
            #                                       id2word=dictionary,
            #                                       num_topics=i,
            #                                       iterations=100,
            #                                       alpha='auto',
            #                                       random_state=100,
            #                                       update_every=1,
            #                                       chunksize=10,
            #                                       passes=20,
            #                                       per_word_topics=True)
            lda = ldamulticore.LdaMulticore(corpus=self.corpus_tfidf,
                                            id2word=self.dictionary,
                                            passes=20,
                                            num_topics=i,
                                            workers=4,
                                            iterations=100,
                                            alpha='symmetric',
                                            gamma_threshold=0.001)
            coh_model_lda = CoherenceModel(model=lda, corpus=self.corpus_tfidf, dictionary=self.dictionary, coherence='u_mass')
            coherence_value = coh_model_lda.get_coherence()

            # coh = lda.log_perplexity(corpus)
            coherence_list.append(coherence_value)

            print('k = {}  coherence value = {}'.format(str(i), str(coherence_value)))

        # for co_value in coherence_list:
        df = pd.DataFrame({'num': com_nums, 'co_value':coherence_list})
        delta = df['co_value'].diff() / df['co_value'][1:]
        df['delta'] = df['co_value'].diff()
        find = df['delta'] == df['delta'].max()
        df_find = df[find]
        optimal_value = 0
        if coherence_list[0] >= df_find['delta'].tolist()[0]:
            optimal_value = coherence_list[0]
            optimal_num = com_nums[0]
        else:
            optimal_value = df_find['delta'].tolist()[0]
            optimal_num = df_find['num'].tolist()[0]

        print('==== coherence values =====')
        print(df, end='\n')
        print('==== final values =====')
        print(df_find)

        df.to_csv(self.model_path+self.data_name+'_coherence_delta.csv', mode='w', encoding='utf-8')


        coh_dict = dict(zip(com_nums, coherence_list))
        sorted_coh_dict = sorted(coh_dict.items(), key=operator.itemgetter(1), reverse=True)
        plt.plot(com_nums, coherence_list, marker='o')
        plt.xlabel('topic')
        plt.ylabel('coherence value')
        plt.draw()
        fig = plt.gcf()
        fig.savefig(self.model_path+self.data_name+'_coherence.png')
        t_ind = np.argmax(coherence_list)
        # self.num_topics = sorted_coh_dict[0][0]
        print('optimal topic number = ', optimal_num)
        return optimal_num

    def saveLDAModel(self):
        print(' ...start to build lda model...')
        # dictionary = corpora.Dictionary(self.corpus)
        # corpus = [dictionary.doc2bow(text) for text in self.corpus]
        # tfidf = TfidfModel(corpus)
        # corpus_tfidf = tfidf[corpus]

        lda_model = ldamulticore.LdaMulticore(corpus=self.corpus_tfidf,
                                        id2word=self.dictionary,
                                        passes=20,
                                        num_topics=self.num_topics,
                                        workers=4,
                                        iterations=100,
                                        alpha='symmetric',
                                        gamma_threshold=0.001)
        with open(self.model_path+self.data_name+'_lda_model.pickle', 'wb') as f:
            pickle.dump(lda_model, f)

        all_topics = lda_model.get_document_topics(self.corpus_tfidf, minimum_probability=0.5, per_word_topics=False)

        documents = self.documents

        with open(self.model_path + self.data_name + '_lda.results', 'w', -1, 'utf-8') as f:
            for doc_idx, topic in enumerate(all_topics):
                    print(doc_idx, ' || ', topic)
                    if len(topic) == 1:
                        topic_id, prob = topic[0][0], topic[0][1]
                        f.writelines(str(doc_idx) + "\u241E" + documents[doc_idx].strip() + "\u241E" + ' '.join(self.corpus[doc_idx]) + "\u241E" + str(topic_id) + "\u241E" + str(prob) + '\n')
        lda_model.save(self.model_path + self.data_name +'_lda.model')
        with open(self.model_path+self.data_name+'_model.dictionary', 'wb') as f:
            pickle.dump(self.dictionary, f)

        return lda_model

    def main(self):
        # self.model_path = self.make_save_path('models/0722')
        self.saveLDAModel()

class LDAEvaluator:
    def __init__(self, config):
        self.data_path = config.data_path
        self.model_path = config.tm_model_path
        self.data_name = config.data_file_name
        self.data = pd.read_csv(self.data_path + self.data_name + '.csv', encoding='utf-8')
        self.all_topics = self.load_results(result_fname=self.model_path + self.data_name + '_lda.results')
        self.model = LdaModel.load(self.model_path + self.data_name + '_lda.model')
        self.corpus = self.get_corpus(self.data_path + 'data_result/' + self.data_name + '.corpus')
        self.corpus_tfidf = self.get_corpus(self.data_path + 'data_result/' + self.data_name + '.corpus_tfidf')
        self.dictionary = self.get_dictionary(self.model_path + self.data_name + '_lda.model.id2word')
        self.topic_num = self.get_topic_num()

    def get_topic_terms(self):
        topics = []
        for i in range(0, self.model.topic_num):
            topic = self.model.show_topic_words(i)
            topics.append(topic)
        topic_terms = []
        topic_values = []
        for topic in topics:
            each_terms = []
            each_values = []
            for term in topic:
                each_terms.append(term[0])
                each_values.append(term[1])
            topic_terms.append(each_terms)
            topic_values.append(each_values)

        df_term = pd.DataFrame(topic_terms)
        df_value = pd.DataFrame(topic_values)
        df_term.to_csv(self.model_path + self.data_name + '_lda_term.csv', mode='w', encoding='utf-8')
        df_value.to_csv(self.model_path + self.data_name + '_lda_value.csv', mode='w', encoding='utf-8')

    def get_topic_num(self):
        df = pd.read_csv(self.model_path+self.data_name+'_coherence_delta.csv', encoding='utf-8')
        max_value = df['delta'] == df['delta'].max()
        topic_num = df[max_value]
        return topic_num['num'].tolist()[0]

    def view_lda_model(self, model, corpus, dictionary):
        # corpus = [dictionary.doc2bow(doc) for doc in corpus]
        prepared_data = gensimvis.prepare(model, corpus, dictionary, mds='mmds')
        pyLDAvis.save_json(prepared_data, self.model_path+self.data_name+'_vis_result.json')
        pyLDAvis.save_html(prepared_data, self.model_path+self.data_name+'_vis_result.html')

    def get_corpus(self, corpus_file):
        with open(corpus_file, 'rb') as f:
            corpus = pickle.load(f)
        return corpus

    def load_results(self, result_fname):
        topic_dict = defaultdict(list)
        patent_titles = self.data['patentTitle'].tolist()
        with open(result_fname, 'r', encoding='utf-8') as f:
            for line in f:
                patent_id, sentence, _, topic_id, prob = line.strip().split('\u241E')
                topic_dict[int(topic_id)].append((patent_titles[int(patent_id)], float(prob)))

        for key in topic_dict.keys():
            topic_dict[key] = sorted(topic_dict[key], key=lambda x: x[1], reverse=True)
        return topic_dict

    def show_topic_docs(self, topic_id, topn=10):
        print(len(self.all_topics[25]))
        return self.all_topics[topic_id][:topn]

    def show_topic_words(self, topic_id, topn=10):
        # print(self.model.show_topic(topic_id, len(self.corpus)))
        return self.model.show_topic(topic_id, 20)

    def show_document_topics(self):
        topics = self.model.get_document_topics(self.corpus_tfidf, minimum_probability=0.5, per_word_topics=False)
        ids = self.data['patentTitle'].tolist()
        topic_li = []
        for idx, topic in enumerate(topics):
            if topic != []:
                topic_li.append(str(topic[0][0]))
            else:
                topic_li.append('NA')

        df = pd.DataFrame({"id" : ids, 'topic_id': topic_li})
        df.to_csv(self.model_path+self.data_name+'.distribution', mode='w', encoding='utf-8')
        topic_set = set(topic_li)
        print(topic_set)
        frequency = {}
        for t in topic_li:
            if t in frequency:
                frequency[t] += 1
            elif t not in frequency:
                frequency[t] = 1
        print(frequency)
        plt.xlabel('Topic')
        plt.ylabel('Frequency')
        plt.grid(False)

        sorted_Dict_Values = sorted(frequency.values(), reverse=True)
        sorted_Dict_Keys = sorted(frequency, key=frequency.get, reverse=True)

        plt.bar(range(len(frequency)), sorted_Dict_Values, align='center')
        plt.xticks(range(len(frequency)), list(sorted_Dict_Keys),  fontsize=5)
        # plt.figure(num=None, figsize=(20, 10), dpi=80)
        # plt.tight_layout()
        # plt.show()
        plt.draw()
        plt.savefig(self.model_path+self.data_name+'_distribution.png', dpi=300, bbox_inches='tight')

        plt.show()

    def show_topics(self, model):
        return self.model.show_topics(fommated=False)

    def get_dictionary(self, dic_fname):
        with open(dic_fname, 'rb') as f:
            dictionary = pickle.load(f)
        return dictionary
