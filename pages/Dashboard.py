import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

def header():
    st.markdown("""
            <div style="display:flex;flex-direction:column;align-items:center;margin-bottom:30px">
                <h1 style="font-size:5rem">NLP Dashboard</h1>
                <div style="width:100%;display:flex;align-items:center;justify-content:center;background-color:#3C3D37;padding:20px 20px;border-radius:10px;font-size:1rem;">
                    <text style="text-align: center">Here is the NLP Dashboard of your data</text>
                </div>
            </div>
        """, unsafe_allow_html=True)

def tab_menu():
    df = pd.read_csv("data/cleaned-data.csv")

    data_overview_tab, word_cloud_tab, word_frequency_tab, clustering_tab, sentiment_analysis_tab = st.tabs(
        ["Data Overview", "Word Cloud", "Word Frequency", "Clustering", "Sentiment Analysis"],
    )

    with data_overview_tab:
        data_overview(df)

    with word_cloud_tab:
        word_cloud(df)

    with word_frequency_tab:
        word_frequency(df)

    with clustering_tab:
        clustering(df)

    with sentiment_analysis_tab:
        sentiment_analysis(df)

def data_overview(df):
    st.write("## Data Preview")
    st.dataframe(df.head())

def word_cloud(df):
    st.write("### Word Cloud")
    text_data = df['text'].tolist()
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(" ".join(text_data))
    fig, ax = plt.subplots()
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)

def word_frequency(df):
    st.write("### Word Frequency")
    text_data = df['text'].tolist()
    tokens = " ".join(text_data).split()
    freq = pd.Series(tokens).value_counts()
    fig, ax = plt.subplots()
    sns.barplot(x=freq[:10].index, y=freq[:10].values, ax=ax)
    ax.set_xlabel('Words')
    ax.set_ylabel('Frequency')
    st.pyplot(fig)

def clustering(df):
    pass

def sentiment_analysis(df):
    st.write("### Sentiment Analysis")
    sentiment_counts = df['label'].value_counts()

    st.write("Sentiment Distribution")
    fig, ax = plt.subplots()
    sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, ax=ax)
    ax.set_xlabel('Sentiment')
    ax.set_ylabel('Count')
    st.pyplot(fig)

def main():
    st.set_page_config(initial_sidebar_state="collapsed")
    with open("src/style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    header()
    tab_menu()

if __name__ == "__main__":
    main()
