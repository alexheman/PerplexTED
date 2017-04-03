import pickle, numpy, pandas as pd
from nltk.collocations import BigramCollocationFinder, TrigramCollocationFinder, QuadgramCollocationFinder
from itertools import chain
from nltk.tokenize import word_tokenize
import warnings
from nltk.corpus import stopwords


from os import path
from PIL import Image
import matplotlib.pyplot as plt

from wordcloud import WordCloud



def corpus_ngram(corpus, ngram = 'bigram'):
    if ngram == 'bigram':
        finder = BigramCollocationFinder.from_words(list(chain.from_iterable(corpus)))
    elif ngram == 'trigram':
        finder = TrigramCollocationFinder.from_words(list(chain.from_iterable(corpus)))
    else:
        return
    #QuadgramCollocationFinder

    ngram_freqdist = finder.ngram_fd
    total_ngram_count = sum(ngram_freqdist.itervalues())
    ngram_freqdist['N_count'] = float(total_ngram_count)

    for ngram in ngram_freqdist.iterkeys():
        if ngram != 'N_count':
            ngram_freqdist[ngram] /= ngram_freqdist['N_count']
    return ngram_freqdist



def calculate_perplexity_score(corpus_freq, test_document, ngram = 'bigram'):
    #document should be a list of all the words in order
    if ngram == 'bigram':
        docs = BigramCollocationFinder.from_words(test_document).ngram_fd
    elif ngram == 'trigram':
        docs = TrigramCollocationFinder.from_words(test_document).ngram_fd
    else:
        return

    ''' Laplace Smoothing - Counting unseen ngrams, replacing the original ngram probabilities, adding in the unseen ngram probabilities of 1/(N+V)'''
    updated_corpus_freq = corpus_freq.copy()
    V_count = 0.0
    N_count = updated_corpus_freq['N_count']

    '''Counting the unseen ngrams as V_count'''
    for unseen_ngram_key in docs.iterkeys():
        if updated_corpus_freq.get(unseen_ngram_key) == None:
            V_count += 1
    updated_corpus_freq['V_count'] = V_count

    '''Replacing the original ngram probabilities'''
    for every_ngram_key in updated_corpus_freq.iterkeys():
        if every_ngram_key != 'N_count' and every_ngram_key != 'V_count':
            updated_corpus_freq[every_ngram_key] = (updated_corpus_freq[every_ngram_key] + 1.0) / (N_count + V_count)

    '''adding in the unseen ngram probabilities of 1/(N+V)'''
    unseen_ngram_probability = 1.0 / (N_count + V_count)
    for unseen_ngram_key in docs.iterkeys():
        if updated_corpus_freq.get(unseen_ngram_key) == None:
            updated_corpus_freq[unseen_ngram_key] = unseen_ngram_probability

    '''Calculating Perplexity Score'''
    score = 0
    for ngram_key in docs.iterkeys():
        if updated_corpus_freq.get(ngram_key) != unseen_ngram_probability:
            score += numpy.log(updated_corpus_freq.get(ngram_key))*docs[ngram_key]
        else:
            score += numpy.log(unseen_ngram_probability)
    return score*(-1.0/(len(test_document)-1))

    #overflow error so we log it
    # score = 1.0
    # for key in docs:
    #     if corpus_freq.get(key) != None:
    #         score *= (1.0/corpus_freq.get(key))
    # return score**(-1.0/len(docs))

def calculate_all_perplexity_scores(corpus, name, dataframe):
    corpus_bigram_percents = corpus_ngram(corpus, 'bigram')
    dataframe['bigram_perplexity_scores'] = [calculate_perplexity_score(corpus_bigram_percents, x, 'bigram') for x in dataframe['full_transcript']]

    corpus_trigram_percents = corpus_ngram(corpus, 'trigram')
    dataframe['trigram_perplexity_scores'] = [calculate_perplexity_score(corpus_bigram_percents, y, 'trigram') for y in dataframe['full_transcript']]

    # pickle_data(name, dataframe)


def pickle_data(filename, df):
    fileObject = open(filename,'wb')
    pickle.dump(df, fileObject)
    fileObject.close()


if __name__ == '__main__':
    fileObject = open('data/nyt_all_words_corpus_data', 'r')
    corpus = pickle.load(fileObject)
    gg = corpus_ngram(corpus)
    for hi in gg.most_common(150):
        print hi
    # fileObject2 = open('data/new_partial_NMF_clusters', 'r')
    # NMF_cluster = pickle.load(fileObject2)
    # fileObject3 = open('data/new_partial_LDA_clusters', 'r')
    # LDA_cluster = pickle.load(fileObject3)
    #
    #
    # calculate_all_perplexity_scores(corpus, 'data/new_perplexity_scores_NMF_clusters', NMF_cluster)
    # calculate_all_perplexity_scores(corpus, 'data/new_perplexity_scores_LDA_clusters', LDA_cluster)
    #
    #
    # warnings.simplefilter(action = "ignore", category = FutureWarning)
    #
    # fileObject4 = open('data/perplexity_scores_NMF_clusters', 'r')
    # NMF_cluster = pickle.load(fileObject4)
    #
    # NMF_cluster = NMF_cluster.sort(columns = 'bigram_perplexity_scores', ascending = True)[:-10]
    #
    # NMF_0 = NMF_cluster[NMF_cluster['NMF_cluster']==0].sort(columns = 'bigram_perplexity_scores', ascending = True)
    #
    # NMF_1 = NMF_cluster[NMF_cluster['NMF_cluster']==1].sort(columns = 'bigram_perplexity_scores', ascending = True)
    #
    # NMF_2 = NMF_cluster[NMF_cluster['NMF_cluster']==2].sort(columns = 'bigram_perplexity_scores', ascending = True)
    #
    # NMF_3 = NMF_cluster[NMF_cluster['NMF_cluster']==3].sort(columns = 'bigram_perplexity_scores', ascending = True)
    #
    # NMF_4 = NMF_cluster[NMF_cluster['NMF_cluster']==4].sort(columns = 'bigram_perplexity_scores', ascending = True)
    #
    # NMF_5 = NMF_cluster[NMF_cluster['NMF_cluster']==5].sort(columns = 'bigram_perplexity_scores', ascending = True)
    #
    # NMF_6 = NMF_cluster[NMF_cluster['NMF_cluster']==6].sort(columns = 'bigram_perplexity_scores', ascending = True)
    # NMF_7 = NMF_cluster[NMF_cluster['NMF_cluster']==7].sort(columns = 'bigram_perplexity_scores', ascending = True)
    # NMF_8 = NMF_cluster[NMF_cluster['NMF_cluster']==8].sort(columns = 'bigram_perplexity_scores', ascending = True)
    # NMF_9 = NMF_cluster[NMF_cluster['NMF_cluster']==9].sort(columns = 'bigram_perplexity_scores', ascending = True)
    #
    #
    # stop = stopwords.words('english')
    #
    # additional = set(['youre', 'ive', 'dont', 'something', 'could', 'really', 'little', 'actually', 'lot', 'look', 'come', 'theyre', 'us','also','go','get'])
    #
    # for term in stop:
    #     additional.add(term.encode('ascii'))
    #
    # alice_mask = numpy.array(Image.open("test.jpg"))
    #
    #
    # ''' experience Life happiness'''
    # # good2 = NMF_5['matrix_corpus'][3]
    # # wc = WordCloud(background_color="white", max_words=2000, mask=alice_mask, stopwords = additional, random_state = 5)
    #
    # ''' girl
    # good1 = NMF_1['matrix_corpus'][0]
    # wc = WordCloud(background_color="white", max_words=2000, mask=alice_mask, stopwords = additional, random_state = 5)'''
    #
    # ''' child love'''
    # text = NMF_6['matrix_corpus'][3]
    # wc = WordCloud(background_color="white", max_words=2000, mask=alice_mask, stopwords = additional, random_state = 3)
    #
    # ''' neuron cells
    # text = NMF_9['matrix_corpus'][3]
    # wc = WordCloud(background_color="white", max_words=2000, mask=alice_mask, stopwords = additional, random_state = 4)'''
    #
    # # wc = WordCloud(background_color="white", max_words=2000, mask=alice_mask, stopwords = additional, random_state = 4)
    # wc.generate(text)
    #
    # plt.imshow(wc, interpolation='bilinear')
    # plt.axis("off")
    # plt.show()
