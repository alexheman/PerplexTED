import os, re, pymongo as mongogod, pandas as pd, numpy as np, pickle
from itertools import chain

from collections import Counter, defaultdict
from nltk import FreqDist, pos_tag
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer



'''######################################################'''
'''Part 1: Reading in all files'''

'''Read all files that end with ".stm"'''
def read_all_files_in_folder(collection, folder):
    path = 'data/{}'.format(folder)
    for ted_talk in os.listdir(path):
        if ted_talk.endswith('.stm'):
            read_file(collection, '{}/{}'.format(path,ted_talk))

'''Read file - reads one file and inserts it into mongodb'''
def read_file(collection, file):
    with open(file) as f:
        author = os.path.basename(file)
        ted_talk_content = f.read().split("\n")
        insert_contents(collection, ted_talk_content, author)

def insert_contents(collection, content_text, author):
    dic = {'raw_content':content_text, 'author':author}
    collection.insert_one(dic)

'''######################################################'''
'''Part 2: Cleaning, joining, and inserting all documents to create a raw_content and cleaned_content key'''


'''Cleans and joins all lines from each Ted Talk to create a combined cleaned_content version'''
def get_clean_talks(collection):
    for transcript in collection.find():
        t_id = transcript['_id']
        split_ted_talk = []
        for line in transcript['raw_content']:
            if len(line) > 1:
                useful_line_piece = line.split('>')[1]
                if not useful_line_piece.startswith(' ignore_'):
                    split_ted_talk.append(transform_text(useful_line_piece))
        full_ted_talk = ' '.join(split_ted_talk)
        collection.update_one({'_id': t_id},{'$set': {'cleaned_content':full_ted_talk}})

'''Removes extra spaces to the right and left of each line. Removes apostrophies'''
def transform_text(text):
    no_extra_space = text.rstrip().lstrip()
    no_unicode = no_extra_space.encode('ascii',errors='ignore')
    no_apostrophes = re.sub(r"\'s",' is', no_unicode)
    no_apostrophes2 = re.sub(r'n\'t',' not', no_apostrophes)
    no_apostrophes3 = re.sub(r'\'m',' am', no_apostrophes2)
    no_apostrophes4 = re.sub(r'\'d',' would', no_apostrophes3)
    no_apostrophes5 = re.sub(r" '",'', no_apostrophes4)
    no_apostrophes6 = re.sub(r"'",'', no_apostrophes5)
    return no_apostrophes6

'''Cleans Author Title Name to create author_clean'''
def make_author_pretty(title):
    first_split = title.split('_')
    hi = ' '.join(re.findall('[A-Z][a-z]*', first_split[0]))
    year = first_split[1][0:4]
    return ' '.join([hi,year])

def fix_author_list(collection):
    for transcript in collection.find():
        t_id = transcript['_id']
        collection.update_one({'_id': t_id},{'$set': {'author_clean': make_author_pretty(transcript['author'])}})


'''######################################################'''
'''Part 3: Tokenization, Removal of Stop Words, and Stemmer'''

'''Tokenize words in each document'''
def tokenize_words(document):
    tokenized_document = word_tokenize(document)
    return tokenized_document

'''Remove stop words from each document'''
def stop_words_remover(document):
    stop = set(stopwords.words('english'))
    stop_words_removed_document = [word for word in document if word not in stop]
    return stop_words_removed_document

'''Checks wordnet Tag for the Lemmatizer'''
def get_wordnet(tag):
    if tag.startswith('NNP'):
        return 'Remove'
    elif tag.startswith('CD'):
        return 'Remove'
    elif tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    else:
        return wordnet.ADJ

'''Lems all words'''
def lem_words(document):
    wordnet = WordNetLemmatizer()
    lemmatized_document = []

    for word in document:
        check_word = word[0].lower()
        tag = get_wordnet(word[1])
        if tag != 'Remove':
            lemmatized_document.append(str(wordnet.lemmatize(check_word, tag)))
    return lemmatized_document

def check(collection):
    transcript = collection.find()[126]
    t_id = transcript['_id']
    tokenized_document = tokenize_words(transcript['cleaned_content'])
    print tokenized_document
    perplexity_lemmatized_document = lem_words(pos_tag(tokenized_document))


def make_squeaky_clean(collection):
    for transcript in collection.find():
        t_id = transcript['_id']
        tokenized_document = tokenize_words(transcript['cleaned_content'])
        with_stop_words_removed = stop_words_remover(tokenized_document)
        clustering_lemmatized_document = lem_words(pos_tag(with_stop_words_removed))
        cluster_words = ' '.join(clustering_lemmatized_document)
        # clustering is for making clusters
        perplexity_lemmatized_document = lem_words(pos_tag(tokenized_document))
        perplexity_words = ' '.join(perplexity_lemmatized_document)
        # perplexity is for perplexity score used later

        collection.update_one({'_id': t_id},{'$set': {'cluster_words': cluster_words}})
        collection.update_one({'_id': t_id},{'$set': {'perplexity_words': perplexity_words}})


def pickle_data(filename, collection, key):
    fileObject = open(filename,'wb')
    list_of_list_of_words = [doc[key] for doc in collection.find()]
    pickle.dump(list_of_list_of_words, fileObject)
    fileObject.close()

def word_count(list_of_list_of_words):
    c = Counter()
    for list_of_words in list_of_list_of_words:
        c.update(list_of_words)
    fdist = FreqDist(c)
    return fdist.most_common(100)


if __name__ == '__main__':
    client = mongogod.MongoClient()
    db = client['newtedtalks']
    collection = db['contents']

    '''Part 1: Reading in all files'''
    read_all_files_in_folder(collection, 'big_train')

    '''Part 2: Cleaning, joining, and inserting all documents to create a raw_content and cleaned_content key'''
    get_clean_talks(collection)
    fix_author_list(collection)

    '''Part 3'''
    make_squeaky_clean(collection)

    '''The Clustering Matrix Data will be used for clustering while the 'ted_talks_all_words_corpus_data' will be used for the perplexity score calculation'''
    pickle_data('data/new_clustering_matrix_data', collection, 'cluster_words')
    pickle_data('data/new_author_clean_matrix_data', collection, 'author_clean')

    pickle_data('data/new_ted_talks_all_words_corpus_data', collection, 'perplexity_words')

    client.close()
