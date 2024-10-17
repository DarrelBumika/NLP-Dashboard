import pandas as pd
import re
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

factory = StopWordRemoverFactory()
stopword_remover = factory.create_stop_word_remover()

def data_cleaning(sent):
    if sent == 'nan':
        return ''
    string = sent.lower() # Mengubah kata menjadi huruf kecil
    string = re.sub(r'@[\w]+', '', string) # Mengapus mention
    string = re.sub(r'https?://\S+|www\.\S+', '', string) # Menghapus link
    string = re.sub(r'[^a-zA-Z0-9 ]', '', string) # Menghapus emoticon dan tanda baca
    string = re.sub(r'rt', '', string) # Menghapus RT
    string = re.sub(r"premium|zonauang|wtb|wts", "", string) # Menghapus kata kunci
    string = string.strip() # Menghapus spasi di awal dan akhir kalimat
    string = stopword_remover.remove(string) # Menghapus stopwords
    return string

def clean(path):
    document = pd.read_csv(path, sep=',')

    document['text'] = document.text.apply(data_cleaning)
    document = document.dropna(subset=['text'])

    document.to_csv('data/cleaned-data.csv', index=False)
