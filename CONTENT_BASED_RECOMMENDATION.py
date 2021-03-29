#############################
# Content Filtering
#############################

# Kullanıcıların beğendiği ya da satın aldığı ürünlere benzer olan ürünleri
# önermek şeklinde çalışır.

#############################
# Film Overview'larına Göre Tavsiye Sistemi
#############################

# Adım 1: TF-IDF MATRISININ OLUSTURULMASI
# Adım 2: COSINE SIMILARITY MATRISININ OLUSTURULMASI
# Adım 3: BİR FİLME EN BENZER OLAN FİLMLERİ ÖNER


#################################
# 1. TF-IDF MATRISININ OLUSTURULMASI
#################################

import pandas as pd
pd.set_option('display.max_columns', 30)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv("datasets/the_movies_dataset/movies_metadata.csv", low_memory=False)  # DtypeWarning kapamak icin
df.head()
df.shape

df["overview"].head()

# 1: countVectorizer yöntemi
# 2: tf-idf yöntemi

#################################
# 1: countVectorizer yöntemi
#################################


# 1,2,3,...,10000
# 12,23,12,...,0

#################################
# 2: tf-idf yöntemi
#################################

# TF-IDF = TF(t) * IDF(t)
# TF(t) = (Bir t teriminin ilgili dokümanda gözlenme frekansı) / (Dokümandaki toplam terim sayısı) (term frequency)
# IDF(t) = log_e(Toplam doküman sayısı / İçinde t terimi olan doküman sayısı) (inverse document frequency)

df['overview'].head()
df['overview'] = df['overview'].fillna('')

tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['overview'])
tfidf_matrix.shape

df['title'].shape

#################################
# 2. COSINE SIMILARITY MATRISININ OLUSTURULMASI
#################################

cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
cosine_sim.shape
cosine_sim[1]

#################################
# 3. BİR FİLME EN BENZER OLAN FİLMLERİ ÖNER
#################################

df = df[~df["title"].isna()]

indices = pd.Series(df.index, index=df['title'])

indices = indices[~indices.index.duplicated(keep='last')]

indices.shape
indices[:10]

indices["Sherlock Holmes"]

movie_index = indices["Sherlock Holmes"]

cosine_sim[movie_index]

similarity_scores = pd.DataFrame(cosine_sim[movie_index], columns=["score"])

movie_indices = similarity_scores.sort_values("score", ascending=False)[1:11].index

df['title'].iloc[movie_indices]


def content_based_recommender(title, cosine_sim, dataframe):
    # index'leri olusturma
    dataframe = dataframe[~dataframe["title"].isna()]
    indices = pd.Series(dataframe.index, index=dataframe['title'])
    indices = indices[~indices.index.duplicated(keep='last')]
    # title'ın index'ini yakalama
    movie_index = indices[title]
    # title'a gore benzerlik skorlarını hesapalama
    similarity_scores = pd.DataFrame(cosine_sim[movie_index], columns=["score"])
    # kendisi haric ilk 10 filmi getirme
    movie_indices = similarity_scores.sort_values("score", ascending=False)[1:11].index
    return dataframe['title'].iloc[movie_indices]

content_based_recommender("Sherlock Holmes", cosine_sim, df)
content_based_recommender("The Godfather", cosine_sim, df)
content_based_recommender('The Dark Knight Rises', cosine_sim, df)


del cosine_sim

content_based_recommender('The Dark Knight Rises', cosine_sim, df)

def calculate_cosine_sim(dataframe):
    tfidf = TfidfVectorizer(stop_words='english')
    dataframe['overview'] = dataframe['overview'].fillna('')
    tfidf_matrix = tfidf.fit_transform(dataframe['overview'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return cosine_sim


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
df = pd.read_csv("datasets/the_movies_dataset/movies_metadata.csv", low_memory=False)

cosine_sim = calculate_cosine_sim(df)

content_based_recommender('The Dark Knight Rises', cosine_sim, df)

content_based_recommender('Ali Baba and the Forty Thieves', cosine_sim, df)

