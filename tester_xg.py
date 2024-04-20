import pandas as pd
import tensorflow as tf
from keras.models import load_model
from keras.layers import TextVectorization
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import numpy as np
import xgboost as xgb

# Loading the data
data = pd.read_csv('FIT1043-MusicGenre-Submission.csv')
data = data.dropna()

# Scaling and normalizing the data
df = data.copy()

loaded_model = xgb.Booster()
loaded_model.load_model("xgb_model.json")

# vectorise text data into int
artist_name_vectorizer = TextVectorization(output_mode='int')
artist_name_vectorizer.adapt(df['artist_name'])
artist_name_vectorized = artist_name_vectorizer(df['artist_name'])

# flatten
artist_name_vectorized = tf.reduce_mean(artist_name_vectorized, axis=-1)

df['artist_name'] = artist_name_vectorized.numpy()

track_name_vectorizer = TextVectorization(output_mode='int')
track_name_vectorizer.adapt(df['track_name'])
track_name_vectorized = track_name_vectorizer(df['track_name'])

# flatten
track_name_vectorized = tf.reduce_mean(track_name_vectorized, axis=-1)
df['track_name'] = track_name_vectorized.numpy()

# Seperating features and the label
features = df.drop(columns=['instance_id'])

# debug 
print(f'features columns: {features.columns}')

sclr = StandardScaler()
features = pd.DataFrame(sclr.fit_transform(features), columns=features.columns)

dtest = xgb.DMatrix(features)

predictions = loaded_model.predict(dtest)

class_labels = np.argmax(predictions, axis=1)

# Step 5: Add the predictions to the dataframe
df['music_genre'] = class_labels

df['instance_id'] = data['instance_id']
final_df = df[['instance_id', 'music_genre']]

# Step 6: Write the dataframe to a new CSV file
final_df.to_csv('gg_pred.csv', index=False)