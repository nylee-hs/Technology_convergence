from DataInput import Configuration, DataInput
from topicModeling import LDABuilder, LDAEvaluator
import pandas as pd
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def main():
    config = Configuration()
    while(True):
        print('=============== MENU ====================')
        print('==== 1. Preprocessing(only)          ====')
        print('==== 2. Topic Modeling(ALL)          ====')
        print('==== 3. Topic Modeling(build)        ====')
        print('==== 4. Topic Modeling(evaluation)   ====')
        print('==== 5. All                          ====')
        print('==== 6. Get Documents by Topic ID    ====')
        print('==== 7. END                          ====')
        print('=========================================')

        choice = input('  >> select number:  ')

        if choice == '1':
            inputdata =DataInput(config)

        elif choice == '2':
            builder = LDABuilder(config=config)
            builder.num_topics = builder.getOptimalTopicNum()
            # builder.num_topics = 30

            builder.main()

            model = LDAEvaluator(config=config)
            topics = []
            for i in range(builder.num_topics):
                topic = model.show_topic_words(i)
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
            # print(topic_terms)
            # model.show_document_topics()
            # topic_number = [f'Topic_{i}' for i in range(len(topic_terms))]
            # terms = [terms[0] for terms in model.show_topic_words(0)]
            # values = [terms[1] for terms in model.show_topic_words(0)]
            # print(values)
            #
            df_term = pd.DataFrame(topic_terms)
            df_value = pd.DataFrame(topic_values)
            df_term.to_csv(config.tm_model_path + config.data_file_name + '_lda_term.csv', mode='w', encoding='utf-8')
            df_value.to_csv(config.tm_model_path + config.data_file_name + '_lda_value.csv', mode='w', encoding='utf-8')
            #
            model.view_lda_model(model.model, model.corpus_tfidf, model.dictionary)

        elif choice == '3':
            builder = LDABuilder(config=config)
            builder.num_topics = builder.getOptimalTopicNum()
            # builder.num_topics = 30

            builder.main()

        elif choice == '4':
            model = LDAEvaluator(config=config)
            print(f'topicNum = {model.topic_num}')
            # topics = []
            # for i in range(0, model.topic_num):
            #     topic = model.show_topic_words(i)
            #     topics.append(topic)
            # topic_terms = []
            # topic_values = []
            # for topic in topics:
            #     each_terms = []
            #     each_values = []
            #     for term in topic:
            #         each_terms.append(term[0])
            #         each_values.append(term[1])
            #     topic_terms.append(each_terms)
            #     topic_values.append(each_values)
            # print(topic_terms)
            model.show_document_topics()
            # topic_number = [f'Topic_{i}' for i in range(len(topic_terms))]
            # terms = [terms[0] for terms in model.show_topic_words(0)]
            # values = [terms[1] for terms in model.show_topic_words(0)]
            # print(values)
            #
            # df_term = pd.DataFrame(topic_terms)
            # df_value = pd.DataFrame(topic_values)
            # df_term.to_csv(config.tm_model_path + config.data_file_name + '_lda_term.csv', mode='w', encoding='utf-8')
            # df_value.to_csv(config.tm_model_path + config.data_file_name + '_lda_value.csv', mode='w', encoding='utf-8')
            #
            model.view_lda_model(model.model, model.corpus_tfidf, model.dictionary)

        elif choice == '5':
            file_list = ['IoT', '가상현실', '드론', '블록체인', '빅데이터', '웨어러블', '인공지능', '클라우드']
            for file in file_list:
                config = Configuration(filename=file, date='1229')
                inputdata = DataInput(config)
                builder = LDABuilder(config=config)
                builder.num_topics = builder.getOptimalTopicNum()
                # builder.num_topics = 30

                builder.main()

                model = LDAEvaluator(config=config)
                topics = []
                for i in range(builder.num_topics):
                    topic = model.show_topic_words(i)
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
                # print(topic_terms)
                # model.show_document_topics()
                # topic_number = [f'Topic_{i}' for i in range(len(topic_terms))]
                # terms = [terms[0] for terms in model.show_topic_words(0)]
                # values = [terms[1] for terms in model.show_topic_words(0)]
                # print(values)
                #
                df_term = pd.DataFrame(topic_terms)
                df_value = pd.DataFrame(topic_values)
                df_term.to_csv(config.tm_model_path + config.data_file_name + '_lda_term.csv', mode='w',
                               encoding='utf-8-sig')
                df_value.to_csv(config.tm_model_path + config.data_file_name + '_lda_value.csv', mode='w',
                                encoding='utf-8-sig')
                #
                model.view_lda_model(model.model, model.corpus_tfidf, model.dictionary)

        elif choice == '6':
            model = LDAEvaluator(config=config)
            topic_id = input(' >>> input topic ID: ')
            patents = model.show_topic_docs(topic_id=int(topic_id), topn=10)
            print(patents)

        elif choice == '7':
            break

if __name__=='__main__':
    main()