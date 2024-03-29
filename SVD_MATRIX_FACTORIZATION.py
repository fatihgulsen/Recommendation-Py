############################################
# Matrix Factorization
#############################################

# pip install surprise
import pandas as pd
from surprise import Reader, SVD, Dataset, accuracy
from surprise.model_selection import GridSearchCV, train_test_split, cross_validate

pd.set_option('display.max_columns', None)


#############################
# Veri Setinin Hazırlanması
#############################

movie = pd.read_csv('datasets/movie_lens_dataset/movie.csv')
rating = pd.read_csv('datasets/movie_lens_dataset/rating.csv')
df = movie.merge(rating, how="left", on="movieId")
df.head()

# takip edilebirlik açısından 4 tane film seçiyorum:
movie_ids = [130219, 356, 4422, 541]
movies = ["The Dark Knight (2011)",
          "Cries and Whispers (Viskningar och rop) (1972)",
          "Forrest Gump (1994)",
          "Blade Runner (1982)"]


sample_df = df[df.movieId.isin(movie_ids)]
sample_df.shape
sample_df.head()

# user movie df'inin oluşturulması
user_movie_df = sample_df.pivot_table(index=["userId"], columns=["title"], values="rating")
user_movie_df.shape
user_movie_df.head()


# rating'lerin scale'ini ifade ediyoruz.
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(sample_df[['userId', 'movieId', 'rating']], reader)

##############################
# MODELLEME
##############################

trainset, testset = train_test_split(data, test_size=.25)
svd_model = SVD()
svd_model.fit(trainset)
predictions = svd_model.test(testset)

accuracy.rmse(predictions)

cross_validate(svd_model, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

user_movie_df

# userid 1 için blade runner tahmini yapalım: 541
svd_model.predict(uid=1.0, iid=541, verbose=True)

# userid 1 için Whispers tahmini yapalım: 356
svd_model.predict(uid=1.0, iid=356, verbose=True)



##############################
# MODEL TUNING
##############################

# GridSearchCV
param_grid = {'n_epochs': [5, 10], 'lr_all': [0.002, 0.005]}

gs = GridSearchCV(SVD, param_grid, measures=['rmse', 'mae'], cv=3, n_jobs=-1, joblib_verbose=True)

gs.fit(data)

gs.best_score['rmse']
# 0.9300

gs.best_params['rmse']
# {'n_epochs': 10, 'lr_all': 0.002}


##############################
# Final model ve Prediction
##############################

svd_model = SVD(**gs.best_params['rmse'])
data = data.build_full_trainset()
svd_model.fit(data)

# userid 1 için blade runner tahmini yapalım: 541
svd_model.predict(uid=1.0, iid=541, verbose=True)

# userid 1 için Whispers tahmini yapalım: 356
svd_model.predict(uid=1.0, iid=356, verbose=True)

# https://surprise.readthedocs.io/en/stable/getting_started.html











