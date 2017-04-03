import os, re, pymongo as mongogod, pandas as pd, numpy as np, pickle
from itertools import chain
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def create_TFIDF_vector(corpus, stop_list):
    '''Creates a TFIDF Vector and corresponding word matrix from the selected corpus and stop list'''
    vectorizer = TfidfVectorizer(max_df = 0.75, max_features = 2000, min_df = 0.05, stop_words = stop_list, ngram_range = (1,2))
    tfidf_vector = vectorizer.fit_transform(corpus)
    word_matrix = vectorizer.get_feature_names()
    return tfidf_vector, word_matrix

def create_count_vector(corpus, stop_list):
    ''' Creates a Count Vector and corresponding word matrix from the selected corpus and stop list'''
    vectorizer = CountVectorizer(max_df = 0.8, max_features = 2000, min_df = 0.03, stop_words = stop_list, ngram_range = (1,2))
    count_vector = vectorizer.fit_transform(corpus)
    word_matrix = vectorizer.get_feature_names()
    return count_vector, word_matrix

def create_NMF_model(vector, word_matrix):
    '''Creates an NMF model using selected vector. Calls upon get_top_words to provide the top 30 words per cluster'''
    nmf = NMF(n_components = 10,
                max_iter = 150, alpha = 1)
    nmf.fit(vector)
    topics_words = nmf.components_
    '''Remove'''
    print("Topics in NMF model:")
    NMF_top_words = get_top_words(nmf, word_matrix)
    z = nmf.transform(vector)
    zcount = Counter([np.argmax(i) for i in z])
    print zcount

    return nmf.transform(vector), NMF_top_words


def create_LDA_model(vector, word_matrix):
    lda = LatentDirichletAllocation(n_topics=10, max_iter= 15, learning_method = 'batch', learning_offset = 50, batch_size = 300)
    lda.fit(vector)
    topics_words = lda.components_
    '''Remove'''
    print("Topics in LDA model:")
    LDA_top_words = get_top_words(lda, word_matrix)
    v = lda.transform(vector)
    vcount = Counter([np.argmax(i) for i in v])
    print vcount

    return lda.transform(vector), LDA_top_words


def get_top_words(model, word_matrix, n_top_words=20):
    '''Returns the top 30 words from each topic cluster'''
    list_of_list_of_top_words =[]
    for index,topic in enumerate(model.components_):
        print("Topic #%d:" % (int(index)))
        print(" ".join([word_matrix[i] for i in topic.argsort()[:-n_top_words - 1:-1]]))
        top_words = [word_matrix[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
        list_of_list_of_top_words.append(top_words)
    return list_of_list_of_top_words


def create_dataframe(vector, word_matrix, author_list, corpus):
    '''Creates a pandas dataframe from the given vector, word matrix, and author list'''
    vector_list = vector.todense().tolist()
    df = pd.DataFrame(vector_list, columns = word_matrix, index = author_list)
    df['matrix_corpus'] = [corpus[x] for x in xrange(len(df))]
    return df

def update_dataframe_clusters(df, name, clusters, top_words, perplexity_list):
    '''Creates three columns for: the cluster group numbering, the top words within the corresponding clusters, and the full transcript of the talk'''
    df[name] = [np.argmax(i) for i in clusters]
    df['top_words'] = [top_words[x] for x in df[name].values]
    df['full_transcript'] = [perplexity_list[x] for x in xrange(len(df))]
    return df

def make_NMF_clusters_df(corpus, stop_list, author_list, perplexity_list):
    '''This creates the full NMF clusters dataframe. The dataframe contains the Author List, the TFIDF vector scores for each word, the NMF cluster group numbering, the top words within the corresponding clusters, and the full transcript for each talk'''
    tfidf_vector, word_matrix = create_TFIDF_vector(corpus, stop_list)
    NMF_clusters, NMF_top_words = create_NMF_model(tfidf_vector, word_matrix)
    NMF_df = create_dataframe(tfidf_vector, word_matrix, author_list, corpus)
    update_dataframe_clusters(NMF_df, 'NMF_cluster', NMF_clusters, NMF_top_words, perplexity_list)
    return NMF_df

def make_LDA_clusters_df(corpus, stop_list, author_list, perplexity_list):
    '''This creates the full LDA clusters dataframe. The dataframe contains the Author List, the Count vector scores for each word, the LDA cluster group numbering, the top words within the corresponding clusters, and the full transcript for each talk'''
    count_vector, word_matrix = create_count_vector(corpus, stop_list)
    LDA_clusters, LDA_top_words = create_LDA_model(count_vector, word_matrix)
    LDA_df = create_dataframe(count_vector, word_matrix, author_list, corpus)
    update_dataframe_clusters(LDA_df, 'LDA_cluster', LDA_clusters, LDA_top_words, perplexity_list)
    return LDA_df


def pickle_df(filename, df):
    fileObject = open(filename,'wb')
    pickle.dump(df, fileObject)
    fileObject.close()

if __name__ == '__main__':

    '''Opening Preprocessed Data'''
    fileObject1 = open('data/new_clustering_matrix_data', 'r')
    TED_words = pickle.load(fileObject1)
    fileObject2 = open('data/new_author_clean_matrix_data', 'r')
    author_list = pickle.load(fileObject2)
    fileObject3 = open('data/new_ted_talks_all_words_corpus_data', 'r')
    perplexity_list = pickle.load(fileObject3)

    '''Defining an additional stop words list'''
    additional_stoplist = ['youre', 'ive', 'dont', 'something', 'could', 'really', 'little', 'actually', 'lot', 'look', 'come', 'theyre', 'us','also']

    '''Creates the NMF cluster dataframe'''
    NMF_df = make_NMF_clusters_df(TED_words, additional_stoplist, author_list, perplexity_list)

    '''Pickling the full and partial NMF dataframes. The partial does not contain the vectors'''
    partial_NMF_clusters = NMF_df.filter(['NMF_cluster','top_words', 'full_transcript', 'matrix_corpus'], axis=1)
    # pickle_df('data/new_full_NMF_cluster', NMF_df)
    # pickle_df('data/new_partial_NMF_clusters', partial_NMF_clusters)


    # '''Creates the LDA cluster dataframe'''
    # LDA_df = make_LDA_clusters_df(TED_words, additional_stoplist, author_list, perplexity_list)
    #
    # '''Pickling the full and partial NMF dataframes. The partial does not contain the vectors'''
    # partial_LDA_clusters = LDA_df.filter(['LDA_cluster','top_words', 'full_transcript', 'matrix_corpus'], axis=1)
    # pickle_df('data/new_full_LDA_cluster', LDA_df)
    # pickle_df('data/new_partial_LDA_clusters', partial_LDA_clusters)
