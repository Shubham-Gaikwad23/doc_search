from django.shortcuts import render
from django.views import View
import pickle
import pandas as pd
import numpy as np
import re
import elasticsearch
from elasticsearch import Elasticsearch
from elasticsearch import helpers


class Search(View):
    def get(self, request, *args, **kwargs):
        return render(request, 'index.html')


class Results(View):
    def get(self, request, *args, **kwargs):
        query = request.GET.get('query')
        results,size1 = self.search_query(query)
        context = {"results": results, "query": query,"size" : size1 }
        return render(request, 'results.html', context=context)
    
    def get_vector(self, query):
        a_file = open("mitten_embedding.pkl", "rb")
        word_embedding = pickle.load(a_file)
        a_file.close()
        vector= sum([word_embedding.get(w, np.zeros((50,))) for w in str(query).split()])/(len(str(query).split())+0.001)
        return vector
        
    def cosine_similarity(self,a, b):
        nominator = np.dot(a, b)

        a_norm = np.sqrt(np.sum(a**2))
        b_norm = np.sqrt(np.sum(b**2))

        denominator = a_norm * b_norm

        cosine_similarity = nominator / denominator

        return cosine_similarity
    
    def preprocess_query(self,document):
        # Remove all the special characters
        document = re.sub(r'\W', ' ', str(document))
        document = re.sub(r'[^A-Za-z0-9]+',' ', document)
        #document= re.sub(r'[^\w\s]',' ', document) 

        # remove all single characters
        document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)

        # Remove single characters from the start
        #document = re.sub(r'\^[a-zA-Z]\s+', ' ', document)

        # Substituting multiple spaces with single space
        document = re.sub(r'\s+', ' ', document, flags=re.I)
        document = document.lower()

        tokens = document.split()
        
        preprocessed_text = ' '.join(tokens)

        return preprocessed_text


    def search_query(self, query: str):
        # TODO: Implement query search by connecting to ML model
        a_file = open("mitten_sentences_df.pkl", "rb")
        Sentence_df = pickle.load(a_file)
        a_file.close()
        #es = Elasticsearch([{'host':'localhost','port': 9200}])
        #es.ping()
        #res = es.search(index='squad_sentences', body='', size=1000)
        #res3 = res['hits']['hits']
        #Sentences = []
        #doc_name = []
        #Sen_vector=[]
        #for i in res3:
        #    doc_name.append(i['_source']['Document_name'])
        #    Sentences.append(i['_source']['Sentences'])
        #    Sen_vector.append(i['_source']['Sentence_vector'])
        #Sentence_df =pd.DataFrame({'Doc_name':doc_name,'Sentences':Sentences,'Sentence_vectors':Sen_vector})
        
        
        query = self.preprocess_query(str(query))
        query_vec = self.get_vector(str(query))
        
        similarity = []
        
        for i in Sentence_df.Sentence_vectors:
            similarity.append(self.cosine_similarity(query_vec,np.asarray(i, dtype='float32')))
        
        # df_results=Sentence_df
        df_results= pd.DataFrame(Sentence_df)                      
        df_results['Similarity_score'] = similarity

        df_results=df_results.sort_values(by = 'Similarity_score',ascending = False)
        df_results=df_results.head(10)
        
        list_of_lists = df_results.values.tolist()
        unique_values = set([list[0] for list in list_of_lists])
        Sentence_punc = [[list[1] for list in list_of_lists if list[0] == value] for value in unique_values]
        Similarity_Score = [[list[3] for list in list_of_lists if list[0] == value] for value in unique_values]
        
        new_df = pd.DataFrame([unique_values,Sentence_punc,Similarity_Score]).transpose()
        new_df.columns=['Doc_name','Sentences_punc','Similarity_Score']
        l = []
        for i in new_df.Similarity_Score:
            l.append(max(i))
        new_df['max_score'] = l
        new_df = new_df.sort_values(by = 'max_score', ascending = False)
                              
        result = []
        title = 'title'
        line = 'line_matches'
        score = 'similarity_score'

        for i,j,k in zip(new_df.Doc_name,new_df.Sentences_punc,new_df.Similarity_Score):
            #print(i,j,k)
            new_dict = {}
            new_dict[title] = i
            new_dict[line] = j
            new_dict[score] = k
            result.append(new_dict)
        size1 = len(result)
        
        return result,size1
                              
    