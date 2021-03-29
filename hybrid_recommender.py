import pandas as pd

movie = pd.read_csv('movie.csv')
rating = pd.read_csv('rating.csv')
df = movie.merge(rating, how="left", on="movieId")

df['title'] = df.title.str.replace('(\(\d\d\d\d\))', '')
df['title'] = df['title'].apply(lambda x: x.strip())
values_title = pd.DataFrame(df["title"].value_counts())
rare_movies = values_title[values_title["title"] <= 1000].index
common_movies = df[~df["title"].isin(rare_movies)]
user_movie_df = common_movies.pivot_table(index=["userId"], columns=["title"], values="rating")
user_id = 108170

random_user_df = user_movie_df[user_movie_df.index == user_id]
random_user_df

movies_watched = random_user_df.columns[random_user_df.notna().any()].tolist()

len(movies_watched)

movies_watched_df = user_movie_df[movies_watched]
movies_watched_df.head()
movies_watched_df.shape

user_movie_count = movies_watched_df.T.notnull().sum()
user_movie_count = user_movie_count.reset_index()
user_movie_count.columns = ["userId", "movie_count"]

perc = len(movies_watched) * 60 / 100
users_same_movies = user_movie_count[user_movie_count["movie_count"] > perc]["userId"]

final_df = pd.concat([movies_watched_df[movies_watched_df.index.isin(users_same_movies.index)],
                      random_user_df[movies_watched]])

final_df.head()
final_df.T.corr()
final_df.shape

corr_df = final_df.T.corr().unstack().sort_values().drop_duplicates()
corr_df = pd.DataFrame(corr_df, columns=["corr"])
corr_df.index.names = ['user_id_1', 'user_id_2']
corr_df = corr_df.reset_index()
corr_df.head()


top_users = corr_df[(corr_df["user_id_1"] == user_id) & (corr_df["corr"] >= 0.65)][
    ["user_id_2", "corr"]].reset_index(drop=True)

top_users = top_users.sort_values(by='corr', ascending=False)

top_users.rename(columns={"user_id_2": "userId"}, inplace=True)

top_users

top_users_ratings = top_users.merge(rating[["userId", "movieId", "rating"]], how='inner')
top_users_ratings
top_users_ratings.shape

top_users_ratings['weighted_rating'] = top_users_ratings['corr'] * top_users_ratings['rating']
top_users_ratings.head()

temp = top_users_ratings.groupby('movieId').sum()[['corr', 'weighted_rating']]
temp.columns = ['sum_corr', 'sum_weighted_rating']

temp.head()

recommendation_df = pd.DataFrame()
recommendation_df['weighted_average_recommendation_score'] = temp['sum_weighted_rating'] / temp['sum_corr']
recommendation_df['movieId'] = temp.index
recommendation_df = recommendation_df.sort_values(by='weighted_average_recommendation_score', ascending=False)
recommendation_df.head(10)

movie_user = movie.loc[movie['movieId'].isin(recommendation_df.head(10)['movieId'].head())]['title']
movie_user.head()
movie_user[:5].values



df['year_movie'] = df.title.str.extract('(\(\d\d\d\d\))', expand=False) #4 değer olan ifadeyi çek
df['year_movie'] = df.year_movie.str.extract('(\d\d\d\d)', expand=False) #parantezlerin içine alıyoruz
df['title'] = df.title.str.replace('(\(\d\d\d\d\))', '')  #title içindeki yılı temizliyoruz
df['title'] = df['title'].apply(lambda x: x.strip()) #oluşan  boşulkları sil

df.shape  #(3400256, 7)
df.head()


df["genre"] = df["genres"].apply(lambda x: x.split("|")[0])
df.drop("genres", inplace=True, axis=1)
df.head()


df.info()  #timestamp time formatında olması lazım

df["timestamp"] = pd.to_datetime(df["timestamp"], format='%Y-%m-%d')
df.info() #datetime64[ns]

#Verileri ayrı ayrı çektik
df["year"] = df["timestamp"].dt.year
df["month"] = df["timestamp"].dt.month
df["day"] = df["timestamp"].dt.day
df.head()

df.shape
df["title"].nunique() #eşsiz film sayısı :26213
a = pd.DataFrame(df["title"].value_counts())
a.head() #titlelara gelen puanlar

rare_movies = a[a["title"] <= 1000].index  #1000 yorumun altındaki filmleri filtreledik
common_movies = df[~df["title"].isin(rare_movies)]
common_movies.shape #(2059083, 10)
common_movies["title"].nunique()  #859

item_movie_df = common_movies.pivot_table(index=["userId"], columns=["title"], values="rating")
item_movie_df.shape  #(23149, 859)
user_movie_df.head(10)
item_movie_df.columns

len(item_movie_df.columns)
common_movies["title"].nunique()

movieId = rating[(rating["rating"] == 5.0) & (rating["userId"] ==user_id)].sort_values(by="timestamp",ascending=False)["movieId"][0:1].values[0]
movie_title = movie[movie["movieId"] == movieId]["title"].str.replace('(\(\d\d\d\d\))', '').str.strip().values[0]

movie = item_movie_df[movie_title]
movie_item = item_movie_df.corrwith(movie).sort_values(ascending=False)
movie_item = item_movie_df[1:6].index


data_user_item = pd.DataFrame()
data_user_item["user_recommendations"] = movie_user[:5].values.tolist()
data_user_item["item_recommendations"] = movie_item[1:6].index
data_user_item
