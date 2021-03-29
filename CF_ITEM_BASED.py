###########################################
# Item-Based Collaborative Filtering (Item-Item Filtering)
###########################################

# Adım 1: Veri Setinin Hazırlanması
# Adım 2: User Movie Df'inin Oluşturulması
# Adım 3: Korelasyona Dayalı Item-Based Film Önerilerinin Yapılması
# Adım 4: İşlemlerin Fonksiyonlaştırılması

######################################
# Adım 1: Veri Setinin Hazırlanması
######################################

# Puan verilme alışkanlıkları birbirine benzer olan filmler üzerinden tavsiye sistemi geliştirmek.

import pandas as pd
pd.set_option('display.max_columns', 20)

movie = pd.read_csv('datasets/movie_lens_dataset/movie.csv')
rating = pd.read_csv('datasets/movie_lens_dataset/rating.csv')
df = movie.merge(rating, how="left", on="movieId")
df.head()

#################
# title
#################

df['year_movie'] = df.title.str.extract('(\(\d\d\d\d\))', expand=False)
df['year_movie'] = df.year_movie.str.extract('(\d\d\d\d)', expand=False)
df['title'] = df.title.str.replace('(\(\d\d\d\d\))', '')
df['title'] = df['title'].apply(lambda x: x.strip())

df.shape
df.head()

#################
# genres
#################

df["genre"] = df["genres"].apply(lambda x: x.split("|")[0])
df.drop("genres", inplace=True, axis=1)
df.head()

#################
# timestamp
#################

df.info()

df["timestamp"] = pd.to_datetime(df["timestamp"], format='%Y-%m-%d')
df.info()

df["year"] = df["timestamp"].dt.year
df["month"] = df["timestamp"].dt.month
df["day"] = df["timestamp"].dt.day
df.head()

######################################
# Adım 2: User Movie Df'inin Oluşturulması
######################################

df.shape
df["title"].nunique()
a = pd.DataFrame(df["title"].value_counts())
a.head()

rare_movies = a[a["title"] <= 1000].index
common_movies = df[~df["title"].isin(rare_movies)]
common_movies.shape
common_movies["title"].nunique()

user_movie_df = common_movies.pivot_table(index=["userId"], columns=["title"], values="rating")
user_movie_df.shape
user_movie_df.head(10)
user_movie_df.columns

len(user_movie_df.columns)
common_movies["title"].nunique()

######################################
# Adım 3: Korelasyona Dayalı Item-Based Film Önerilerinin Yapılması
######################################

movie = user_movie_df["Matrix, The"]

user_movie_df.corrwith(movie).sort_values(ascending=False).head(10)


######################################
# 4. Adım: İşlemlerin Fonksiyonlaştırılması
######################################




def create_user_movie_df():
    import pandas as pd
    movie = pd.read_csv('datasets/movie_lens_dataset/movie.csv')
    rating = pd.read_csv('datasets/movie_lens_dataset/rating.csv')
    df = movie.merge(rating, how="left", on="movieId")
    df['title'] = df.title.str.replace('(\(\d\d\d\d\))', '')
    df['title'] = df['title'].apply(lambda x: x.strip())
    a = pd.DataFrame(df["title"].value_counts())
    rare_movies = a[a["title"] <= 1000].index
    common_movies = df[~df["title"].isin(rare_movies)]
    user_movie_df = common_movies.pivot_table(index=["userId"], columns=["title"], values="rating")
    return user_movie_df


user_movie_df = create_user_movie_df()


######################################
# USER-MOVIE DF'inin Daha Sonra Kullanılmak Üzere Saklanması ve Daha Sonra Çağırılması
######################################

import pickle
pickle.dump(user_movie_df, open("user_movie_df.pkl", 'wb'))

import pickle
import pandas as pd
user_movie_df = pickle.load(open('user_movie_df.pkl', 'rb'))


movie = user_movie_df["Matrix, The"]
user_movie_df.corrwith(movie).sort_values(ascending=False).head(10)

movie_name = pd.Series(user_movie_df.columns).sample(1).values[0]
movie = user_movie_df[movie_name]
user_movie_df.corrwith(movie).sort_values(ascending=False).head(10)


def item_based_recommender(movie_name, user_movie_df):
    movie = user_movie_df[movie_name]
    return user_movie_df.corrwith(movie).sort_values(ascending=False).head(10)

item_based_recommender("Matrix, The", user_movie_df)
item_based_recommender("Sherlock Holmes", user_movie_df)


# GENELLEME PROBLEMI

item_based_recommender("Matrix", user_movie_df)



def item_based_recommender(movie_name):
    # film umd'de yoksa önce ismi barındıran ilk filmi getir.
    # eger o da yoksa filmin isminin ilk iki harfini barındıran ilk filmi getir.
    if movie_name not in user_movie_df:
        # ismi barındıran ilk filmi getir.
        if [col for col in user_movie_df.columns if movie_name.capitalize() in col]:
            new_movie_name = [col for col in user_movie_df.columns if movie_name.capitalize() in col][0]
            movie = user_movie_df[new_movie_name]
            print(F"{movie_name}'i barındıran ilk  film: {new_movie_name}\n")
            print(F"{new_movie_name} için öneriler geliyor...\n")
            return user_movie_df.corrwith(movie).sort_values(ascending=False).head(10)
        # filmin ilk 2 harfini barındıran ilk filmi getir.
        else:
            new_movie_name = [col for col in user_movie_df.columns if col.startswith(movie_name.capitalize()[0:2])][0]
            movie = user_movie_df[new_movie_name]
            print(F"{movie_name}'nin ilk 2 harfini barındıran ilk film: {new_movie_name}\n")
            print(F"{new_movie_name} için öneriler geliyor...\n")
            return user_movie_df.corrwith(movie).sort_values(ascending=False).head(10)
    else:
        print(F"{movie_name} için öneriler geliyor...\n")
        movie = user_movie_df[movie_name]
        return user_movie_df.corrwith(movie).sort_values(ascending=False).head(10)


item_based_recommender("Sherlock Holmes")

item_based_recommender("Matrix, The")

item_based_recommender("Matrix")

item_based_recommender("matrix")

item_based_recommender("ads")

