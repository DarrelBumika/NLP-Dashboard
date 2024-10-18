import streamlit as st
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import stanza

stanza.download('id')
nlp = stanza.Pipeline('id')

# Streamlit app layout
st.title("NLP Dashboard")
st.sidebar.title("Upload your CSV")

# Upload the file
df = pd.read_csv("data/cleaned-data.csv")

# Check for necessary headers
if 'text' in df.columns:
    text_data = df['text']

    st.write("## Data Preview")
    st.dataframe(df.head())

    # Tokenization example
    tokens = [token.text for doc in nlp.pipe(text_data.astype(str)) for token in doc if not token.is_stop]

    # Choose what visualization to display
    option = st.sidebar.selectbox(
        'Choose an analysis:',
        ['Word Cloud', 'Word Frequency', 'Sentiment Analysis']
    )

    if option == 'Word Cloud':
        # 1. Word Cloud Visualization
        st.write("### Word Cloud")
        wordcloud = WordCloud(width=800, height=400, random_state=21, max_font_size=110).generate(" ".join(tokens))
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis('off')
        st.pyplot(plt)

    elif option == 'Word Frequency':
        # 2. Word Frequency Visualization
        st.write("### Word Frequency")
        freq = Counter(tokens)
        common_words = freq.most_common(10)
        df_freq = pd.DataFrame(common_words, columns=["Word", "Frequency"])
        fig, ax = plt.subplots()
        sns.barplot(x='Frequency', y='Word', data=df_freq, ax=ax)
        st.pyplot(fig)

    elif option == 'Sentiment Analysis':
        # 3. Sentiment Analysis Visualization (example using TextBlob)
        from textblob import TextBlob

        st.write("### Sentiment Analysis")
        sentiments = text_data.apply(lambda x: TextBlob(str(x)).sentiment.polarity)
        st.write("Sentiment Distribution")
        fig, ax = plt.subplots()
        sns.histplot(sentiments, bins=20, ax=ax, kde=True)
        st.pyplot(fig)
