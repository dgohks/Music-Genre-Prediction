import pandas as pd
import tensorflow as tf
from keras.models import load_model
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import CountVectorizer


# Loading dataset 
train = pd.read_csv('FIT1043-MusicGenre-Dataset.csv')
data = pd.read_csv('FIT1043-MusicGenre-Submission.csv')
data = data.dropna()
train = train.dropna()
train['name_combined'] = train['artist_name'] + " " + train['track_name']

# Scaling and normalizing the data
df = data.copy()
df = df.drop(columns = ['instance_id'])
df['duration'] = df['duration_ms'].apply(lambda x: round(x/1000))
df['popularity_valence'] = df['popularity'] * df['valence']
df['danceability_energy'] = df['danceability'] * df['energy']
df['acousticness_instrumentalness'] = df['acousticness'] * df['instrumentalness']
df['loudness_energy'] = df['loudness'] * df['energy']
df['speechiness_liveness'] = df['speechiness'] * df['liveness']
df['tempo_energy'] = df['tempo'] * df['energy']
df['name_combined'] = df['artist_name'] + " " + df['track_name']

vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(train['name_combined'])
X_pred = vectorizer.transform(df['name_combined'])

# Seperating features and the label
features = df.drop(columns=['artist_name', 'track_name', 'name_combined', 'duration_ms'])

# Convert the sparse matrix to a DataFrame
df_bow = pd.DataFrame(X_pred.toarray(), columns=vectorizer.get_feature_names_out())

# Normalize
sclr = StandardScaler()
features = pd.DataFrame(sclr.fit_transform(features), columns=features.columns)

# Reset the index of your original DataFrame
features.reset_index(drop=True, inplace=True)

# Concatenate the original DataFrame with the bag of words DataFrame
features = pd.concat([features, df_bow], axis=1)

# Step 3: Load the keras model
model = load_model('wewin_model copy.keras')


# Step 4: Use the model to predict the music genre
predictions = model.predict(features)
class_labels = np.argmax(predictions, axis=1)

# Step 5: Add the predictions to the dataframe
df['music_genre'] = class_labels

df['instance_id'] = data['instance_id']
final_df = df[['instance_id', 'music_genre']]

# Step 6: Write the dataframe to a new CSV file
final_df.to_csv('33521247-Derek Goh Kai Shen-v1.csv', index=False)
