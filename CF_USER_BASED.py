############################################
# User-Based Collaborative Filtering (User-User Filtering)
#############################################

# Kullanıcıların davranış benzerlikleri üzerinden film önerileri yapılacak.


# Adım 1: Veri Setinin Hazırlanması
# Adım 2: Öneri yapılacak kullanıcının izlediği filmlerin belirlenmesi
# Adım 3: Aynı filmleri izleyen diğer kullanıcıların verisine ve id'lerine erişmek
# Adım 4: Öneri yapılacak kullanıcı ile en benzer kullanıcıların belirlenmesi
# Adım 5: Weighted rating'lerin  hesaplanması
# Adım 6: Weighted average recommendation score'un hesaplanması

#############################################
# Adım 1: Veri Setinin Hazırlanması
#############################################


import pandas as pd
from helpers.helpers import create_user_movie_df
user_movie_df = create_user_movie_df()

# import pickle
# user_movie_df = pickle.load(open('user_movie_df.pkl', 'rb'))
# user_movie_df.head()

random_user = int(pd.Series(user_movie_df.index).sample(1, random_state=45).values)


#############################################
# Adım 2: Öneri yapılacak kullanıcının izlediği filmlerin belirlenmesi
#############################################

random_user_df = user_movie_df[user_movie_df.index == random_user]
random_user_df

movies_watched = random_user_df.columns[random_user_df.notna().any()].tolist()

user_movie_df.loc[user_movie_df.index == random_user, user_movie_df.columns == "Schindler's List"]

len(movies_watched)


#############################################
# Adım 3: Aynı filmleri izleyen diğer kullanıcıların verisine ve id'lerine erişmek
#############################################

pd.set_option('display.max_columns', 5)

movies_watched_df = user_movie_df[movies_watched]
movies_watched_df.head()
movies_watched_df.shape

user_movie_count = movies_watched_df.T.notnull().sum()
user_movie_count = user_movie_count.reset_index()
user_movie_count.columns = ["userId", "movie_count"]

user_movie_count[user_movie_count["movie_count"] > 20].sort_values("movie_count", ascending=False)

user_movie_count[user_movie_count["movie_count"] == 33].count()

users_same_movies = user_movie_count[user_movie_count["movie_count"] > 20]["userId"]

#############################################
# Adım 4: Öneri yapılacak kullanıcı ile en benzer kullanıcıların belirlenmesi
#############################################

# Bunun için 3 adım gerçekleştireceğiz:
# 1. Sinan ve diğer kullanıcıların verilerini bir araya getireceğiz.
# 2. Korelasyon df'ini oluşturacağız.
# 3. En benzer bullanıcıları (Top Users) bulacağız.


final_df = pd.concat([movies_watched_df[movies_watched_df.index.isin(users_same_movies.index)],
                      random_user_df[movies_watched]])

final_df.head()

final_df.T.corr()

corr_df = final_df.T.corr().unstack().sort_values().drop_duplicates()
corr_df = pd.DataFrame(corr_df, columns=["corr"])
corr_df.index.names = ['user_id_1', 'user_id_2']
corr_df = corr_df.reset_index()
corr_df.head()


top_users = corr_df[(corr_df["user_id_1"] == random_user) & (corr_df["corr"] >= 0.65)][
    ["user_id_2", "corr"]].reset_index(drop=True)

top_users = top_users.sort_values(by='corr', ascending=False)

top_users.rename(columns={"user_id_2": "userId"}, inplace=True)

top_users

rating = pd.read_csv('datasets/movie_lens_dataset/rating.csv')
top_users_ratings = top_users.merge(rating[["userId", "movieId", "rating"]], how='inner')

top_users_ratings


#############################################
# Adım 5: Weighted rating'lerin  hesaplanması
#############################################

top_users_ratings['weighted_rating'] = top_users_ratings['corr'] * top_users_ratings['rating']
top_users_ratings.head()


#############################################
# Adım 6: Weighted average recommendation score'un hesaplanması
#############################################

temp = top_users_ratings.groupby('movieId').sum()[['corr', 'weighted_rating']]
temp.columns = ['sum_corr', 'sum_weighted_rating']

temp.head()

recommendation_df = pd.DataFrame()
recommendation_df['weighted_average_recommendation_score'] = temp['sum_weighted_rating'] / temp['sum_corr']
recommendation_df['movieId'] = temp.index
recommendation_df = recommendation_df.sort_values(by='weighted_average_recommendation_score', ascending=False)
recommendation_df.head(10)

recommendation_df


movie = pd.read_csv('datasets/movie_lens_dataset/movie.csv')
movie.loc[movie['movieId'].isin(recommendation_df.head(10)['movieId'])]


#############################################
# Fonksiyonlaştırma:
#############################################


def user_based_recommender():
    import pickle
    import pandas as pd
    user_movie_df = pickle.load(open('user_movie_df.pkl', 'rb'))
    random_user = int(pd.Series(user_movie_df.index).sample(1, random_state=45).values)
    random_user_df = user_movie_df[user_movie_df.index == random_user]

    movies_watched = random_user_df.columns[random_user_df.notna().any()].tolist()
    movies_watched_df = user_movie_df[movies_watched]
    user_movie_count = movies_watched_df.T.notnull().sum()
    user_movie_count = user_movie_count.reset_index()
    user_movie_count.columns = ["userId", "movie_count"]
    users_same_movies = user_movie_count[user_movie_count["movie_count"] > 20]["userId"]

    final_df = pd.concat([movies_watched_df[movies_watched_df.index.isin(users_same_movies.index)],
                          random_user_df[movies_watched]])
    corr_df = final_df.T.corr().unstack().sort_values().drop_duplicates()
    corr_df = pd.DataFrame(corr_df, columns=["corr"])
    corr_df.index.names = ['user_id_1', 'user_id_2']
    corr_df = corr_df.reset_index()
    top_users = corr_df[(corr_df["user_id_1"] == random_user) & (corr_df["corr"] >= 0.65)][
        ["user_id_2", "corr"]].reset_index(drop=True)

    top_users = top_users.sort_values(by='corr', ascending=False)
    top_users.rename(columns={"user_id_2": "userId"}, inplace=True)
    rating = pd.read_csv('datasets/movie_lens_dataset/rating.csv')
    top_users_ratings = top_users.merge(rating[["userId", "movieId", "rating"]], how='inner')
    top_users_ratings['weighted_rating'] = top_users_ratings['corr'] * top_users_ratings['rating']
    temp = top_users_ratings.groupby('movieId').sum()[['corr', 'weighted_rating']]
    temp.columns = ['sum_corr', 'sum_weighted_rating']

    recommendation_df = pd.DataFrame()
    recommendation_df['weighted_average_recommendation_score'] = temp['sum_weighted_rating'] / temp['sum_corr']
    recommendation_df['movieId'] = temp.index
    recommendation_df = recommendation_df.sort_values(by='weighted_average_recommendation_score', ascending=False)

    movie = pd.read_csv('datasets/movie_lens_dataset/movie.csv')
    return movie.loc[movie['movieId'].isin(recommendation_df.head(10)['movieId'])]

user_based_recommender()


