import re
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
import pickle

factory = StopWordRemoverFactory()
stopword_remover = factory.create_stop_word_remover()

model = pickle.load(open('model/model-nb.p', 'rb'))
feature_bow = pickle.load(open('model/feature-bow.p', 'rb'))

def data_cleaning(sent):
    if sent == 'nan':
        return ''
    string = sent.lower()  # Mengubah kata menjadi huruf kecil
    string = re.sub(r'@[\w]+', '', string)  # Mengapus mention
    string = re.sub(r'https?://\S+|www\.\S+', '', string)  # Menghapus link
    string = re.sub(r'[^a-zA-Z0-9 ]', '', string)  # Menghapus emoticon dan tanda baca
    string = re.sub(r'rt', '', string)  # Menghapus RT
    string = re.sub(r"premium|zonauang|wtb|wts", "", string)  # Menghapus kata kunci
    string = string.strip()  # Menghapus spasi di awal dan akhir kalimat
    string = stopword_remover.remove(string)  # Menghapus stopwords
    return string

def data_labeling(sent):
    X = feature_bow.transform([sent])
    sentimentLabel = model.predict(X)[0]
    return sentimentLabel

def clean(df):
    df['text'] = df.text.apply(data_cleaning)
    df.dropna(subset=['text'])
    return df

def label(df):
    df['label'] = df.text.apply(data_labeling)
    return df

def save_data(data, path):
    data.to_csv(path, index=False)
