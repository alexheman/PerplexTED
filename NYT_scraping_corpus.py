import pymongo as mongogod
import requests, bs4, time, re, pickle
from pymongo.errors import DuplicateKeyError
from nltk.tokenize import word_tokenize
from nltk import FreqDist, pos_tag
from nltk.corpus import wordnet
from nltk.stem.wordnet import WordNetLemmatizer


'''Part 1: Scraping Articles off the NYT API'''

def single_query(link, payload):
    response = requests.get(link, params=payload)
    if response.status_code != 200:
        print 'WARNING', response.status_code
    else:
        return response.json()

def scrape_article(total_pages, collection):
    link = 'http://api.nytimes.com/svc/search/v2/articlesearch.json'
    payload = {'api-key': '589e63fd7b344806b1a216d4fd809b75','begin_date': "20150101",'end_date': "20170301"}
    for day in range(0,total_pages):
        payload['page'] = str(day)
        content = single_query(link, payload)
        for i in content['response']['docs']:
            try:
                collection.insert_one(i)
                print 'collected!'
            except DuplicateKeyError:
                print 'DUPS!'
        time.sleep(2)

'''Updates the mongo database with the article word contents by using the scraped html link'''
def get_article_content(collection):
    scrape_article(1500, collection)
    links = collection.find({},{'web_url': 1})

    for uid_link in links:
        counter +=1
        if counter % 10 == 0:
            print 'Count: ', counter, ' '
        uid = uid_link['_id']
        link = uid_link['web_url']
        html = requests.get(link).content
        soup = bs4.BeautifulSoup(html, 'html.parser')

        article_content = '\n'.join([i.get_text() for i in soup.find_all('p', class_= "story-body-text story-content")])
        print article_content

        collection.update_one({'_id': uid}, {'$set': {'content_txt': article_content}})


'''Part 2: Cleaning the scraped article content using similar methods used for the TED talks for consistency'''

'''Wordnet Tagging helps with the lemmatization'''
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

'''Lemmatizes each word dependant on the word's wordnet tag'''
def lem_words(document):
    lemmatizer = WordNetLemmatizer()
    lemmatized_document = []
    for word in document:
        check_word = word[0].lower()
        tag = get_wordnet(word[1])
        if tag != 'Remove':
            lemmatized_document.append(str(lemmatizer.lemmatize(check_word, tag)))
    return lemmatized_document

def first_scrub(text):
    text = text.split('___')[0]
    no_bracket_signatures = re.sub(r'.*?\((.*?)\)', '', text)
    no_unicode = no_bracket_signatures.encode('ascii',errors='ignore')
    no_apostrophes = re.sub(r"\'s ",' is ', no_unicode)
    no_apostrophes2 = re.sub(r'n\'t',' not', no_apostrophes)
    no_apostrophes3 = re.sub(r'\'m',' am', no_apostrophes2)
    no_apostrophes4 = re.sub(r'\'d',' would', no_apostrophes3)
    no_symbols = re.sub(r'[^\w]',' ', no_apostrophes4)
    return no_symbols


'''Cleans and lemmatizes the words for every scraped NYT article within the mongo database. Saves and pickles the cleaned version to use in creating the perplexity score'''

def ready_up_and_pickle(collection):
    corpus = []
    for article in collection.find():
        a_id = article['_id']
        cleaned_text = first_scrub(article['content_txt'])
        tokenized_text = word_tokenize(cleaned_text)
        lemmatized_document = lem_words(pos_tag(tokenized_text))
        ready_words = ' '.join(lemmatized_document)

        collection.update_one({'_id': a_id},{'$set': {'ready_words':ready_words}})

        corpus.append(lemmatized_document)
    fileObject = open('data/nyt_all_words_corpus_data','wb')
    pickle.dump(corpus, fileObject)
    fileObject.close()


if __name__ == '__main__':
    client = mongogod.MongoClient()
    db = client['nyt']
    collection = db['corpus']

    get_article_content(client)
    fix(client)
    ready_up_and_pickle(client)

    client.close()
