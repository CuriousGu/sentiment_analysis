from string import punctuation
import unidecode
import nltk


class Text_Model:

    def __init__(self, text, tfidf):
        list_punctuation = [x for x in punctuation]
        stop_words = nltk.corpus.stopwords.words('portuguese')
        irrelevant_data = list_punctuation + stop_words

        self.text = unidecode.unidecode(text).lower()
        self.irrelevant_data = irrelevant_data
        self.tfidf = tfidf
        self.phrase = ' '

    # removing irrelavant infos like pronoumns and applying stemmer
    def adjusting_text(self):

        # removing suffixes
        stemmer = nltk.RSLPStemmer()

        # spliting text and removing punctuation
        punctuation_remover = nltk.tokenize.WordPunctTokenizer()
        self.text = punctuation_remover.tokenize(self.text)

        # removing stopwords
        for word in self.text:
            if word not in self.irrelevant_data:
                word = stemmer.stem(word)
                self.phrase += f'{word} '

        return self

    # applying TF-IDF to measure the relevant of the bigrams
    def measuring_relevance(self):
        self.tfidf_vector = self.tfidf.transform([self.phrase])
        return self

