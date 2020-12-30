from bokeh.models import ColumnDataSource, LabelSet, LinearColorMapper, HoverTool, value
from bokeh.plotting import figure, output_file, save
from bokeh.io import export_png, output_notebook, show
from bokeh.palettes import brewer
from bokeh.transform import factor_cmap
from gensim.models import Doc2Vec
# from sklearn.cluster import KMeans

# from datamanager import DataManager
import random
import pandas as pd
import numpy as np
import pickle
# from visualize_utils import visualize_between_words, visualize_words
from tqdm import tqdm, tnrange
# import pytagcloud
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import os
import multiprocessing
from sklearn.manifold import TSNE
from DataInput import Configuration
# from statsmodels.formula.api import ols
# from statsmodels.stats.anova import anova_lm
# from yellowbrick.cluster import KElbowVisualizer

# import networkx as nx

class Doc2VecModeler:
    def __init__(self, tagged_doc, config):
        self.model_path = config.model_path
        self.data_path = config.data_path
        self.data_name = config.data_file_name
        self.tagged_doc = tagged_doc
        self.model = self.run()

    def run(self):
        print('==== Start Doc2Vec Modeling ====')
        # cores = multiprocessing.cpu_count()
        model = Doc2Vec(self.tagged_doc, dm=0, dbow_words=1, window=10, alpha=0.025, vector_size=1024, min_count=50,
                min_alpha=0.025, workers=4, hs=0, negative=10, epochs=20, sample=0.1, ns_exponent=1e-07)
        model.save(self.model_path + self.data_name + '_doc2vec.model')
        print('==== End Doc2Vec Process ====')
        return model

    # def make_save_path(self): ## directory는 'models/날짜'의 형식으로 설정해야 함
    #     print('==== Start Doc2Vec Process ====')
    #     directory = 'data/doc2vec_test_data/' + input('data date : ') + '/model_doc2vec/'
    #     if not os.path.exists(directory):
    #         os.makedirs(directory)
    #     return directory
    #
    # def get_file_name(self):
    #     file_name = input(' > file_name : ')
    #     return file_name

class Doc2VecEvaluator:

    def __init__(self, config, use_notebook=False):
        self.data_path = config.data_path
        self.model_path = config.model_path
        self.data_name = config.data_file_name

        self.model = Doc2Vec.load(self.model_path+self.data_name+'_doc2vec.model')
        self.doc2idx = {el:idx for idx, el in enumerate(self.model.docvecs.doctags.keys())}
        self.use_notebook = use_notebook

        self.data = pd.read_csv(self.data_path+self.data_name+'.csv', encoding='utf-8')
        self.size = len(self.doc2idx.values())


    # def get_data_name(self):
    #     data_name = input(' > data_name for doc2vec : ') ## 분석하고자 하는 csv 파일의 이름을 입력
    #     return data_name
    #
    # def make_save_path(self): ## directory는 'models/날짜'의 형식으로 설정해야 함
    #     print('==== Analyzing Doc2Vec Process ====')
    #     analysis_date = input(' > date : ')
    #     data_directory = 'data/doc2vec_test_data/'+ analysis_date + '/data/'
    #     model_directory = 'data/doc2vec_test_data/' + analysis_date + '/model_doc2vec/'
    #     if not os.path.exists(data_directory):
    #         os.makedirs(data_directory)
    #     if not os.path.exists(model_directory):
    #         os.makedirs(model_directory)
    #     return data_directory, model_directory

    def get_words(self, corpus_file):
        with open(corpus_file, 'rb') as f:
            data = pickle.load(f)
        return data

    def most_similar_terms(self, topn=10):
        df = pd.DataFrame()
        data = pd.read_csv('analysis/doc2vec_test_data/0702/including_words_list.csv', encoding='utf-8')
        seed_term = data['Includingwords']

        for term in seed_term:
            similar_terms = self.model.wv.most_similar(term)
            temp = []
            for s_term, score in similar_terms:
                if score >= 0.8:
                    temp.append(s_term)
                else:
                    temp.append('none')
            df.loc[:, term] = pd.Series(temp)
        df.to_csv(self.model_path+self.data_name+'_terms_results.csv', mode='w', encoding='utf-8')
        return df

    def most_similar(self, job_id, topn=10):
        similar_jobs = self.model.docvecs.most_similar(f'{self.data_name}_' + str(job_id), topn=topn)
        temp = f'{self.data_name}_' + str(job_id)
        print(f'(Query Job Title : {self.get_job_title(temp)})')
        job_ids = []
        job_titles = []
        scores = []
        for job_id, score in similar_jobs:
            job_titles.append(self.get_job_title(job_id)[0])
            job_ids.append(job_id)
            scores.append(score)
        df = pd.DataFrame({'ID':job_ids, 'Job_Title':job_titles, 'Score':scores})
        df.to_csv(self.model_path+self.data_name+'_sim_matrix.csv', mode='w', encoding='utf-8')
        return df

    def get_types(self, job_id):
        data = self.data
        job_id = str(job_id)
        job_id = job_id.split('_')
        job_id = int(job_id[2])
        s = data[data['id'] == job_id]['type']
        s = s.tolist()
        # s = data[data['id']==int(job_id)]['job_title']
        # print(s)
        return s

    def get_types_in_corpus(self, n_sample=5):
        job_ids = self.model.docvecs.doctags.keys()
        return {job_id: self.get_types(job_id) for job_id in job_ids}

    def get_titles_in_corpus(self, n_sample=5):
        job_ids = self.model.docvecs.doctags.keys()
        # job_ids = random.sample(self.model_doc2vec.docvecs.doctags.keys(), n_sample)
        return {job_id: self.get_job_title(job_id) for job_id in job_ids}

    # def get_word_cloud(self, word_count_dict):
    #     taglist = pytagcloud.make_tags(word_count_dict.items(), maxsize=100)
    #     pytagcloud.create_tag_image(taglist, self.model_path+self.data_name+'_word_cloud.jpg', size=(1200, 800), rectangular=False)

    def get_word_graph(self, word_count_dict):
        plt.xlabel('Word')
        plt.ylabel('Frequency')
        plt.grid(True)


        Sorted_Dict_Values = sorted(word_count_dict.values(), reverse=True)
        Sorted_Dict_Keys = sorted(word_count_dict, key=word_count_dict.get, reverse=True)

        plt.bar(range(len(word_count_dict)), Sorted_Dict_Values, align='center')
        plt.xticks(range(len(word_count_dict)), list(Sorted_Dict_Keys), rotation='90', fontsize=5)
        plt.figure(num=None, figsize=(20, 10), dpi=80)
        plt.show()

    def get_word_count(self, data):
        sline = [' '.join(line) for line in data]
        word_list = []
        for line in sline:
            for word in line.split():
                word_list.append(word)
        word_count = {}
        for word in word_list:
            if word in word_count:
                word_count[word] += 1
            else:
                word_count[word] = 1
        word_count_list = sorted(word_count.items(), key=lambda x:x[1], reverse=True)

        return word_count, word_count_list

    def most_similar_result(self, data_size, topn):
        print('   -> creating most similar job matrix')
        df = pd.DataFrame()
        i = 0
        keys_list = list(self.doc2idx.keys())
        for i in tqdm(range(len(keys_list))):
            # job_id = 'Job_ID_' + str(job_id).split('_')[2]

            title = self.get_job_title(keys_list[i])[0]
            title = f'{title}({str(keys_list[i])})'
            similar_jobs = self.model.docvecs.most_similar(keys_list[i], topn=len(keys_list))
            sim_list = []
            for sim_job_id, score in similar_jobs:
                if score >= 0.6:
                    sim_job_titles = self.get_job_title(sim_job_id)[0]
                    sim_job_id = sim_job_id.split('_')[2]
                    input = f'{sim_job_titles}({sim_job_id})'
                    sim_list.append(input)
                else:
                    sim_list.append('')
            # i = i + 1
            df.loc[:, title] = pd.Series(sim_list)

        df.to_csv(self.model_path+self.data_name+'_sim_title_result.csv', mode='w', encoding='utf-8')
        return df

    # def most_similar_result_with_newwork(self, data_size, topn):
    #     print('   -> creating most similar job matrix with network')
    #     df = pd.DataFrame()
    #     G = nx.karate_club_graph()
    #
    #     i = 0
    #     keys_list = list(self.doc2idx.keys())
    #     nodes = [] ## node list
    #     edges = [] ## edge list(튜플의 형태로 저장)
    #
    #     for job_id in keys_list:
    #         node_id = str(job_id).split('_')[2]
    #         job_id = 'Job_ID_' + str(job_id).split('_')[2]
    #
    #         title = self.get_job_title(job_id)[0]
    #         title = f'{title}({str(job_id)})'
    #
    #         similar_jobs = self.model.docvecs.most_similar(job_id, topn=len(keys_list))
    #
    #         sim_list = []
    #         for sim_job_id, score in similar_jobs:
    #             if score >= 0.8:
    #                 nodes.append(node_id)  ## node list
    #                 sim_job_titles = self.get_job_title(sim_job_id)[0]
    #                 sim_job_id = sim_job_id.split('_')[2]
    #                 input = f'{sim_job_titles}({sim_job_id})'
    #                 sim_list.append(input)
    #                 temp_tuple = (node_id, sim_job_id, score)
    #                 edges.append(temp_tuple)
    #
    #             else:
    #                 sim_list.append('')
    #         i = i + 1
    #         df.loc[:, title] = pd.Series(sim_list)
    #
    #     df.to_csv(self.model_path+self.data_name+'_sim_title_result.csv', mode='w', encoding='utf-8')
    #     nodes = set(nodes)
    #     nodes = list(nodes)
    #     print(len(nodes))
    #     print(nodes[:])
    #     print(edges[:])
    #     G.add_nodes_from(nodes)
    #     G.add_weighted_edges_from(edges)
    #
    #     degree = nx.degree(G)
    #     print(degree)
    #     plt.figure(figsize=(20, 10))
    #     graph_pos = nx.spring_layout(G, k=0.42, iterations=17)
    #     nx.draw_networkx_labels(G, graph_pos, font_size=10, font_family='sans-serif')
    #     # nx.draw_networkx_nodes(G, graph_pos, node_size=[ var * 50 for var in degree], cmap='jet')
    #     nx.draw_networkx_edges(G, graph_pos, edge_color='gray')
    #     nx.draw(G, node_size=[100 + v[1] * 100 for v in degree], with_labels=True)
    #     plt.show()
    #     return df

    def get_job_title(self, job_id):
        data = self.data
        job_id = str(job_id)
        job_id = job_id.split('_')
        job_id = int(job_id[2])
        s = data[data['id'] == job_id]['job_title']
        s = s.tolist()
        # s = data[data['id']==int(job_id)]['job_title']
        # print(s)
        return s

    def get_similarity(self, model):
        print('   -> get similarity values')
        keys_list = list(self.doc2idx.keys())
        # keys_list = [key.split('_')[2] for key in keys_list]
        # type_list = self.data['type'].tolist()


        # total_result_dict = []
        # for i in range(size):
        #     result = self.model_doc2vec.docvecs.most_similar('Job_ID_'+str(i), topn=size)
        #     result_klist = [int(item[0].split('_')[2]) for item in result]
        #     result_slist = [item[1] for item in result]
        #     result_dict = {}
        #     for j in range(len(result_klist)):
        #         result_dict[result_klist[j]] = result_slist[j]
        #     total_result_dict.append(result_dict)
        sim_matrix=[]
        # print(total_result_dict[0])
        # for dic in total_result_dict:
        for i in tqdm(range((len(keys_list)))):
            _matrix = []
            for j in keys_list:
                _matrix.append(self.model.docvecs.similarity(keys_list[i], j))
            sim_matrix.append(_matrix)
        np_matrix = np.array(sim_matrix)
        df = pd.DataFrame(np_matrix)
        # col_name = ['Job_ID_'+str(i) for i in keys_list]
        # row_name = ['Job_ID_'+str(i)+'_'+str(j) for i, j in zip(keys_list, type_list)]
        col_name = keys_list
        row_name = keys_list
        df.columns = col_name
        df.index = row_name
        print(df.head())
        df.loc['Similarity_Average', :] = df.mean()
        df.to_csv(self.model_path+self.data_name+'_sim_matrix.csv', mode='w', encoding='utf-8-sig')
        return df


    # def get_clustering(self):
    #     # nc = range(1,51)
    #     # kmeans = [KMeans(n_clusters = i, init = 'k-means++', max_iter=500) for i in nc]
    #     # scores = [kmeans[i].fit(self.model.docvecs.vectors_docs).inertia_ for i in tqdm(range(len(kmeans)))]
    #     # plt.plot(nc, scores, marker='o')
    #     # plt.xlabel('Number of Clusters')
    #     # plt.ylabel('Score')
    #     # plt.title('Elbow Curve')
    #     # plt.show()
    #     model = KMeans()
    #     visualizer = KElbowVisualizer(model, k = (1, 20))
    #     visualizer.fit(self.model.docvecs.vectors_docs)
    #     visualizer.show()
    #     cluster_no = visualizer.elbow_value_
    #     print(f'   --> cluster number = {cluster_no}')
    #     model = KMeans(n_clusters=cluster_no, algorithm='auto')
    #     model.fit(self.model.docvecs.vectors_docs)
    #     job_ids = self.model.docvecs.doctags.keys()
    #     job_titles = [self.get_job_title(jid) for jid in job_ids]
    #     df = pd.DataFrame({'Job_Title': job_titles, 'Cluster': model.labels_})
    #     df.to_csv(self.model_path+self.data_name+'_cluster.csv', mode='w', encoding='utf-8')
    #     with open(self.model_path+self.data_name+'_cluster.model', 'wb') as f:
    #         pickle.dump(model, f)
    #
    #     # new_df = self.data
    #     # new_df['cluster'] = model.labels_
    #     # new_df.to_csv(self.data_path + self.data_name + '_new.csv', mode='w', encoding='utf-8')
    #
    #     return model




    def word_visulize(self, words, vecs, palette="Viridis256", filename="/notebooks/embedding/words.png",
                        use_notebook=False):
        # circle_size = input('     >> circle size : ')
        # text_size = input('     >> font size : ') + 'pt'

        with open(self.model_path+self.data_name+'_cluster.model', 'rb') as f:
            model = pickle.load(f)

        # cluster = [ f'group{cluster}' for cluster in model.labels_.tolist()]
        cluster = model.labels_.tolist()
        tsne = TSNE(n_components=2, perplexity=50, early_exaggeration=10, metric="cosine")
        tsne_results = tsne.fit_transform(vecs)

        df = pd.DataFrame(columns=['x', 'y', 'id', 'cluster'])
        # df['id'] = list(words)
        words = [ word.split('_')[0] for word in words]

        df['x'], df['y'], df['id'], df['cluster'] = tsne_results[:, 0], tsne_results[:, 1], words, cluster
        # df['x'], df['y'] = tsne_results[:, 0], tsne_results[:, 1]
        df = df.fillna('')
        print(df.head())
        # print(ColumnDataSource.from_df(df))
        # source = ColumnDataSource(ColumnDataSource.from_df(df))
        source = ColumnDataSource(ColumnDataSource.from_df(df))
        labels = LabelSet(x="x", y="y", text="cluster", y_offset=8,
                          text_font_size="0pt", text_color="#555555",
                          source=source, text_align='center')

        color_mapper = LinearColorMapper(palette=palette, low=min(df['cluster']), high=max(df['cluster']))
        group_name = [str(category) for category in df['cluster']]

        # print(brewer[["Spectral"][:]])
        # print(b)
        # colors = brewer[["Spectral"][len(df['cluster'].unique())]]
        # colormap = {i: colors[i] for i in df['cluster'].unique()}
        # colors = [colormap[x] for x in df['cluster']]
        # print(colors)
        # df['color'] = colors

        plot = figure(plot_width=1200, plot_height=1200, output_backend="webgl")
        plot.add_tools(
            HoverTool(
                tooltips='@id'
            )
        )
        plot.scatter("x", "y", size=10, source=source, color={'field': 'cluster', 'transform': color_mapper}, line_color=None,
                     fill_alpha=0.8, legend_field='cluster')

        # plot.circle(source=source, x='x', y='y', line_alplha=0.3, fill_alpha=0.2, size=10, fill_color='color', line_color='color')
        plot.add_layout(labels)
        show(plot)
        output_file(self.model_path+self.data_name+'_tsne.html')
        save(plot)

    def word_visulize_group(self, words, vecs, palette="Viridis256", filename="/notebooks/embedding/words.png",
                        use_notebook=False):
        circle_size = input('     >> circle size : ')
        text_size = input('     >> font size : ') + 'pt'
        groups=['A', 'B', 'C', 'D', 'E', 'F']
        group_list =[]
        for word in words:
            if word.split('_')[1] == 'A':
                group_list.append(groups[0])
            elif word.split('_')[1] == 'B':
                group_list.append(groups[1])
            elif word.split('_')[1] == 'C':
                group_list.append(groups[2])
            elif word.split('_')[1] == 'D':
                group_list.append(groups[3])
            elif word.split('_')[1] == 'E':
                group_list.append(groups[4])
            elif word.split('_')[1] == 'F':
                group_list.append(groups[5])

        tsne = TSNE(n_components=2, perplexity=50, early_exaggeration=10, metric='cosine')
        tsne_results = tsne.fit_transform(vecs)

        df = pd.DataFrame(columns=['x', 'y', 'word', 'group'])
        df['x'], df['y'], df['word'], df['group'] = tsne_results[:, 0], tsne_results[:, 1], list(words), group_list
        # df['x'], df['y'] = tsne_results[:, 0], tsne_results[:, 1]
        df = df.fillna('')
        print(df.head())
        # print(ColumnDataSource.from_df(df))
        source = ColumnDataSource(ColumnDataSource.from_df(df))
        labels = LabelSet(x="x", y="y", text="word", y_offset=8,
                          text_font_size=text_size, text_color="#555555",
                          source=source, text_align='center')

        color_mapper = LinearColorMapper(palette=palette, low=min(tsne_results[:, 1]), high=max(tsne_results[:, 1]))

        group_name = ['A', 'B', 'C', 'D', 'E', 'F']
        colors = ['#FF0000', '#FFBB00', '#00D8FF', '#0055FF', '#6600FF', '#000000']
        plot = figure(plot_width=1200, plot_height=1200)
        # plot.scatter("x", "y", size=int(circle_size), source=source, color={'field': 'y', 'transform': color_mapper}, line_color=None,
        #              fill_alpha=0.8)
        plot.scatter("x", "y", size=int(circle_size), source=source, color=factor_cmap('group', colors, group_name),legend_field='group',
                     line_color=None,
                     fill_alpha=0.8)
        plot.add_layout(labels)
        show(plot)
        output_file(self.model_path+self.data_name+'_tsne.html')
        save(plot)

    def visualize_jobs(self, palette='Viridis256', type='between', choice = None):
        print('   -> Visualization Start')
        if choice == None:
            view_type = 'N'
        else:
            view_type = input('    >> Group(Y/N) : ').capitalize()

        job_ids = self.get_titles_in_corpus(n_sample=len(self.model.docvecs.doctags.keys()))
        job_ids_2 = self.get_types_in_corpus(n_sample=len(self.model.docvecs.doctags.keys()))

        #job_titles = [key for key in job_ids.keys()]
        keys_list = [key for key in job_ids.keys()]
        values_list = [value for value in job_ids.values()]

        keys_list_t = [key for key in job_ids_2.keys()]
        values_list_t = [value for value in job_ids_2.values()]

        job_titles = []
        job_types = []

        for i in range(len(keys_list)):
            key = keys_list[i]
            key = key.split('_')[2]
            value = values_list[i][0]
            value = value.split('_')[-1]
            job_titles.append(key+'_'+value)

        for i in range(len(keys_list_t)):
            key = keys_list_t[i]
            key = key.split('_')[2]
            value = values_list_t[i][0]
            # value = value.split('_')[-1]
            job_types.append(key+'_'+str(value))


        #job_titles = self.get_job_title()
        job_vecs = [self.model.docvecs[self.doc2idx[job_id]] for job_id in job_ids.keys()]
        job_vecs_t = [self.model.docvecs[self.doc2idx[job_id]] for job_id in job_ids_2.keys()]

        if view_type == 'Y':
            self.word_visulize_group(job_types, job_vecs_t, palette, use_notebook=self.use_notebook)
        elif view_type == 'N':
            self.word_visulize(job_titles, job_vecs, palette, use_notebook=self.use_notebook)



    def get_group_data(self, job_type):
        raw = pd.read_csv(self.model_path+self.data_name+'_sim_matrix.csv')
        ## 타입 별 데이터의 사이즈 찾기
        filtered_data = raw.filter(like='_'+job_type, axis=1)
        # filtered_data = filtered_data.filter(like='_'+job_type, axis=0)
        # print(filtered_data)
        size = filtered_data.columns.size
        return filtered_data.iloc[:size, :]

    def get_average_value_groupBy(self):
        print('  --> average of similarity ')
        raw = pd.read_csv(self.model_path + self.data_name + '_sim_matrix.csv')
        types = ['A', 'B', 'C', 'D', 'E', 'F']
        sizes = []
        ## 타입 별 데이터의 사이즈 찾기
        size_1 = 0
        for j_type in types:
            filtered_data = raw.filter(like='_' + j_type, axis=1)
            size_2 = filtered_data.columns.size
            size_2 = size_1 + size_2
            sizes.append([size_1, size_2])
            size_1 = size_2
        print(sizes)

        results_data = []
        groups_data = []
        for j_type in tqdm(range(len(types))):
            temp = []
            temp_group = []
            filtered_data = raw.filter(like='_' + types(j_type), axis=1) ## 타입 데이터 전체
            temp.append(types[j_type])
            temp_group.append(types[j_type])
            for i in range(len(types)):
                extracted_data = filtered_data.iloc[sizes[i][0]:sizes[i][1] , : ]
                group_data = extracted_data.mean()
                temp.append(extracted_data.mean().mean())
                temp_group.append(group_data)
            results_data.append(temp)
            groups_data.append(temp_group)
        with open(self.model_path+self.data_name+'.groups_data', 'wb') as f:
            pickle.dump(groups_data, f)
        print(results_data)

    # def anova_test(self):
    #     print('   --> ANOVA Test...')
    #     # with open('data/doc2vec_test_data/1119/model_doc2vec/1120_all.groups_data', 'rb') as f:
    #     with open(self.model_path+self.data_name+'.groups_data', 'rb') as f:
    #         groups_data = pickle.load(f)
    #     groups = [group for group in groups_data]
    #     print(groups[0][2][0])
    #     print(groups[0][1][0])
    #
    #     types = [data[0] for data in groups]
    #     results = []
    #     sim_matrix = []
    #     for i in tqdm(range(len(groups))):
    #         _matrix = []
    #         check_i = 1
    #         base = groups[i][check_i]
    #         base_name = [types[i] for temp in range(len(base))]
    #         df_base = pd.DataFrame({'Value': base, 'Type': base_name})
    #         for j in range(1,7):
    #             type_index = 0
    #             compare = groups[i][j].tolist()
    #             compare_name = [types[j-1] for temp in range(len(compare))]
    #             print(f'BASE = {base_name[0]} - COMPARE = {compare_name[0]}')
    #
    #             df_compare = pd.DataFrame({'Value': compare, 'Type': compare_name})
    #             print(df_base.mean())
    #             print(df_compare.mean())
    #             frames = [df_base, df_compare]
    #             result_df = pd.concat(frames)
    #
    #             model = ols('Value ~ C(Type)', result_df).fit()
    #             # print(anova_lm(model))
    #             # print(anova_lm(model)['PR(>F)'][0])
    #             _matrix.append(anova_lm(model)['PR(>F)'][0])
    #             type_index += 1
    #         sim_matrix.append(_matrix)
    #         check_i += 1
    #     np_matrix = np.array(sim_matrix)
    #     df = pd.DataFrame(np_matrix)
    #     col_name = types
    #     row_name = types
    #     df.columns = col_name
    #     df.index = row_name
    #     print(df.head())















        # results = [[(self.get_group_data(j_type).sum(axis=0)-1)/2] for j_type in types]


        # i = 0
        # for j_type in types:
        #     print(f'{j_type} : {results[i].mean()}')
        #     i += 1



        #     temp = []
        #     for i in range(len(dic)):
        #         temp.append(dic[i])
        #     print(temp)
    # def get_movie_title(self, movie_id):
    #     url = 'http://movie.naver.com/movie/point/af/list.nhn?st=mcode&target=after&sword=%s' % movie_id.split("_")[1]
    #     resp = requests.get(url)
    #     root = html.fromstring(resp.text)
    #     try:
    #         title = root.xpath('//div[@class="choice_movie_info"]//h5//a/text()')[0]
    #     except:
    #         title = ""
    #     return title

# config = Configuration()
# dve = Doc2VecEvaluator(config)
# dve.get_clustering()
# dve.get_average_value_groupBy()
# dve.anova_test()