import streamlit as st
import spacy
import string
import nltk
import os
import gdown
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
import pickle
import unicodedata
import re
from sklearn.metrics.pairwise import cosine_similarity
from translate import Translator
import numpy as np
import num2words

def dummy(doc):
    return doc

# load the models from disk

@st.cache
def download_docTFIDF():
    url = 'https://drive.google.com/uc?id=1Q4bCwKZGCwsEdAxq7VXsCgxNb2WaS0Mj'
    output = 'docTFIDF.pkl'
    gdown.download(url, output, quiet=False) 

def download_queryTFIDF():
    url = 'https://drive.google.com/uc?id=1O5dKp53h_AaJgD9S2_VEVphRnt_LgPTU'
    output = 'queryTFIDF.pkl'
    gdown.download(url, output, quiet=False) 

if (os.path.exists('queryTFIDF.pkl') == False):
    download_queryTFIDF()

if (os.path.exists('docTFIDF.pkl') == False):
    download_docTFIDF()


nb = pickle.load(open('nb_pipeline.pkl', 'rb'))
stop_words = pickle.load(open('stop_words.pkl', 'rb'))
df_topico_resposta = pickle.load(open('topico_resposta.pkl', 'rb'))


f=open('respostas_ministerio.txt','r',errors = 'ignore')
raw=f.read()
raw = raw.lower()# converts to lowercase
sent_tokens = nltk.sent_tokenize(raw)# converts to list of sentences 

punc = string.punctuation.replace('?','') 

def removerAcentosECaracteresEspeciais(palavra):

    # Unicode normalize transforma um caracter em seu equivalente em latin.
    nfkd = unicodedata.normalize('NFKD', palavra)
    palavraSemAcento = u"".join([c for c in nfkd if not unicodedata.combining(c)])

    # Usa expressão regular para retornar a palavra apenas com números, letras e espaço
    return re.sub('[^a-zA-Z0-9? \\\]', '', palavraSemAcento)

def tokeniza_lemmatiza(pergunta):
     return [tok.lemma_ for tok in nlp(pergunta)]
    
def to_lower(token_lemma):
    return [word.lower() for word in token_lemma]

def no_punc(lowered):
    return [word for word in lowered if word not in punc]

def no_accen(no_punc):
    return [removerAcentosECaracteresEspeciais(word) for word in no_punc]

def stopword_removed(no_accen):
    return [word for word in no_accen if word not in stop_words]


lemmer = nltk.stem.WordNetLemmatizer()
#WordNet is a semantically-oriented dictionary of English included in NLTK.
def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]
remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)

def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

def response(user_response):
    robo_response=''
    sent_tokens.append(user_response)
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words=stop_words)
    tfidf = TfidfVec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx=vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if(req_tfidf==0):
        robo_response=robo_response+"Desculpe! Nao te entendi"
        return robo_response
    else:
        robo_response = robo_response+sent_tokens[idx]
        return robo_response

def convert_lower_case(data):
    return np.char.lower(data)

def remove_stop_words(data):
    stop_words = stopwords.words('english')
    words = word_tokenize(str(data))
    new_text = ""
    for w in words:
        if w not in stop_words and len(w) > 1:
            new_text = new_text + " " + w
    return new_text

def remove_punctuation(data):
    symbols = "!\"#$%&()*+-./:;<=>?@[\]^_`{|}~\n"
    for i in range(len(symbols)):
        data = np.char.replace(data, symbols[i], ' ')
        data = np.char.replace(data, "  ", " ")
    data = np.char.replace(data, ',', '')
    return data

def remove_apostrophe(data):
    return np.char.replace(data, "'", "")

def convert_numbers(data):
    tokens = word_tokenize(str(data))
    new_text = ""
    for w in tokens:
        try:
            w = num2words(int(w))
        except:
            a = 0
        new_text = new_text + " " + w
    new_text = np.char.replace(new_text, "-", " ")
    return new_text

def lematiser(data):
    wnl = WordNetLemmatizer()    
    tokens = word_tokenize(str(data))
    new_text = ""
    for w in tokens:
        new_text = new_text + " " + wnl.lemmatize(w)
    return new_text

def preprocess(data):
    data = convert_lower_case(data)
    data = remove_punctuation(data)
    data = remove_apostrophe(data)
    data = remove_stop_words(data)
    data = convert_numbers(data)
    data = lematiser(data)
    return data

def doc_query_similarity(queryTFIDF,docTFIDF, query):
  queryTFIDF = queryTFIDF.transform([query])
  cosine = cosine_similarity(queryTFIDF, docTFIDF).flatten()
  return cosine

#@st.cache
#def calculate_doctfidf(a):
#     docTFIDF = TfidfVectorizer(use_idf=True, smooth_idf=True).fit_transform(df_cord19['abstract'])
#     return docTFIDF

# @st.cache
# def calculate_querytfidf(a):
#     queryTFIDF = TfidfVectorizer().fit(df_cord19['abstract'])
#     return queryTFIDF


# Set page title
st.title('Máquina de Busca no Contexto da Pandemia COVID-19: um estudo de caso para aplicações Q&A')


st.markdown("<br>", unsafe_allow_html=True)
"""
## Equipe Evolution: Leila Fabiola Ferreira e Mateus Cichelero da Silva
## Ciência de Dados 2 - UTFPR, Professor Luiz Celso Gomes Jr

Esta página tem por objetivo apresentar uma aplicação  de Recuperação de Informação, Q&A e NLP no contexto da pandemia de COVID-19.

O usuário poderá entrar com uma dúvida/ questionamento sobre a pandemia através do campo textual e a partir disso: 

1. Será apresentada a resposta sugerida através da aplicação de um modelo supervisionado de classificação multiclasses (Naive Bayes), treinado utilizando dados de respostas oficiais do Ministério da Saúde e Fundação Fio Cruz;

2. Será apresentada a resposta sugerida de um modelo de recuperação de informação que usa como documentos as sentenças tokenizadas do mesmo Corpus de perguntas e respostas anterior e se baseia na distância de cossenos / TF-IDF;

3. Por fim, é apresentada a sugestão de um artigo relacionado à questão, baseando-se na mesma estratégia do ponto 2, mas utilizando como query a pergunta traduzida para o inglês e como documentos uma base de abstracts de mais de 160 mil artigos do CORD-19.
---
"""

### insert data  ###

pergunta = st.text_input('Faça sua pergunta sobre o Coronavírus/Pandemia:')
pergunta_en = ''

if pergunta != '':
    st.subheader('Resposta Classificador Naive Bayes')

    translator= Translator(from_lang='pt',to_lang="en")
    pergunta_en = translator.translate(pergunta)
    nlp = pickle.load(open('nlp_model.pkl', 'rb'))
    entrada = stopword_removed(
        no_accen(
            no_punc(
                to_lower(
                    tokeniza_lemmatiza(pergunta)))))
    del nlp
    topico= nb.predict([entrada])[0]
    st.write(f'A pergunta foi classificada como sendo do tópico: {topico}')
    st.write('Resposta indicada de uma fonte oficial sobre o tema da pergunta:')
    st.write(df_topico_resposta['resposta'][df_topico_resposta['topico']==topico].iloc[0])

if pergunta != '':
    st.subheader('Resposta Modelo de Recuperação de Informação - Corpus FAQ Ministério da Saúde/ Fio Cruz')

    user_response = pergunta.lower()
    st.write(response(user_response))
    sent_tokens.remove(user_response)    

if pergunta_en != '':
    st.subheader('Sugestão de Artigo - CORD-19')

    query = pergunta_en
    #docTFIDF = calculate_doctfidf(1)
    #queryTFIDF = calculate_querytfidf(1)
    docTFIDF = pickle.load(open('docTFIDF.pkl', 'rb'))
    queryTFIDF = pickle.load(open('queryTFIDF.pkl', 'rb'))
    teste = doc_query_similarity(queryTFIDF, docTFIDF, query)
    del docTFIDF
    del queryTFIDF
    df_cord19 = pd.read_csv('dataset2_cord19.csv')
    df_cord19['score'] = teste
    df_cord19 = df_cord19.sort_values(by=['score'], ascending=False)
    st.write('Aproveite para ler o artigo:')
    st.write(df_cord19['title'].iloc[0])
    st.write('Link do artigo sugerido:')
    st.write(df_cord19['url'].iloc[0])
    del df_cord19