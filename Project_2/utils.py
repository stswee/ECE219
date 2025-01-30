from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import homogeneity_score, completeness_score, v_measure_score, adjusted_rand_score, adjusted_mutual_info_score
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
import re, string
import pandas as pd
nltk.download('punkt_tab', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('omw-1.4', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)

# Given function to parse out HTML-related characters
def clean(text):
    text = re.sub(r'^https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
    texter = re.sub(r"<br />", " ", text)
    texter = re.sub(r"&quot;", "\"",texter)
    texter = re.sub('&#39;', "\"", texter)
    texter = re.sub('\n', " ", texter)
    texter = re.sub(' u '," you ", texter)
    texter = re.sub('`',"", texter)
    texter = re.sub(' +', ' ', texter)
    texter = re.sub(r"(!)\1+", r"!", texter)
    texter = re.sub(r"(\?)\1+", r"?", texter)
    texter = re.sub('&amp;', 'and', texter)
    texter = re.sub('\r', ' ',texter)
    clean = re.compile('<.*?>')
    texter = texter.encode('ascii', 'ignore').decode('ascii')
    texter = re.sub(clean, '', texter)
    if texter == "":
        texter = ""
    return texter

# Define custom transformer to clean the dataset
class DocumentPreprocessingTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X):
        return self

    def transform(self, X):
        # Strip all HTML-related artifacts
        X = X.apply(clean)

        # Remove all punctuation and digits
        ref_punct_digits = string.punctuation + string.digits
        X = X.apply(lambda x: x.translate(str.maketrans('', '', ref_punct_digits)))

        # Makes all characters lower-case
        # Note: Technically the CountVectorizer already handles it, but is necessary
        #       for lemmatization.
        X = X.apply(lambda x: x.lower())
        return X

# Define custom transfomer for lemmiatizatino
class LemmatizationPOSTransfomer(BaseEstimator, TransformerMixin):
    def lemmatize_documents(self, doc, wnl):
        """
        Lemmatizes documents (i.e. each entry) and returns the lemmatized
        document as a single string.
        """
        return ' '.join([wnl.lemmatize(w[0], self.get_wordnet_pos(w[1])) for w in pos_tag(word_tokenize(doc))])


    # Nested function for getting part of speech
    def get_wordnet_pos(self, tag):
        """
        Maps POS tags to WordNet POS tags.

        Default to Noun.
        """
        if tag.startswith('J'):  # Adjective
            return wordnet.ADJ
        elif tag.startswith('V'):  # Verb
            return wordnet.VERB
        elif tag.startswith('N'):  # Noun
            return wordnet.NOUN
        elif tag.startswith('R'):  # Adverb
            return wordnet.ADV
        else:
            return wordnet.NOUN

    def fit(self, X):
        return self

    def transform(self, X):
        wnl = WordNetLemmatizer()
        X = X.apply(lambda x: self.lemmatize_documents(x, wnl))
        return X

def evaluate_cluster_model(true_labels, predict_labels):
    """
    Returns the homogenity, completeness, V-measure, adjusted rand index,
    and the adjusted mutual information scores based on the provided 
    true labels and predicted labels.
    """
    eval_df = pd.DataFrame({
        'Homogeneity': [homogeneity_score(true_labels, predict_labels)],
        'Completeness': [completeness_score(true_labels, predict_labels)],
        'V-measure': [v_measure_score(true_labels, predict_labels)],
        'Adjusted Rand Index': [adjusted_rand_score(true_labels, predict_labels)],
        'Adjusted Mutual Information Score': [adjusted_mutual_info_score(true_labels, predict_labels)]
    })
    
    return eval_df

def get_svd_perc_ratio(n_components, df):
    """
    Returns the sum of explained variance rations for a given matrix of data
    with the provided `n_components` components.

    For some reason, multiprocessing won't work unless the function is imported
    from another file... see link below
    https://stackoverflow.com/questions/41385708/multiprocessing-example-giving-attributeerror-in-jupyter-notebook
    """
    return (n_components, sum(TruncatedSVD(n_components=n_components).fit(df).explained_variance_ratio_))