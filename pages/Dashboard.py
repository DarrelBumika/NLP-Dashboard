import os
import random
from collections import Counter
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
import utils.DataProcessing as DataProcessing

colors = ['#3B3030', '#664343', '#795757', '#FFF0D1', '#F8E9A1']

def text_sentiment(text):
    df = pd.DataFrame({"text": [text]})
    df = DataProcessing.clean(df)
    df = DataProcessing.label(df)
    return df

def text_cluster(df, num_clusters):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df['text']).toarray()

    model = KMeans(n_clusters=num_clusters, init='k-means++', max_iter=1000, n_init=num_clusters)
    model.fit(X)

    pca = PCA(n_components=2, random_state=0)
    reduced_features = pca.fit_transform(X)

    df_plot = pd.DataFrame(reduced_features, columns=['PC1', 'PC2'])
    df_plot['Cluster'] = model.predict(X)

    return df_plot

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
    df["text"] = df["text"].astype(str)

    data_overview_tab, word_cloud_tab, word_frequency_tab, clustering_tab, sentiment_analysis_tab = st.tabs(
        ["👁️ Data Overview", "☁️ Word Cloud", "📊 Word Frequency", "🌐 Clustering", "😡 Sentiment Analysis"],
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
    st.markdown(f"""
            <div style="display:flex;justify-content:space-between;align-items:center">
                <div style="display:flex;flex-direction:column;">
                    <text style="font-size:1rem;transform:translateY(15px)">Total Data</text>
                    <text style="font-weight:bold;font-size:3.5rem;">{df.shape[0]}</text>
                </div>
                <div style="width:1px;height:50px;background-color:#3C3D37"></div>
                <div style="display:flex;flex-direction:column;">
                    <text style="font-size:1rem;transform:translateY(15px)">Positive</text>
                    <text style="font-weight:bold;font-size:3.5rem;">{df[df["label"] == "positive"].shape[0]}</text>
                </div>
                <div style="display:flex;flex-direction:column;">
                    <text style="font-size:1rem;transform:translateY(15px)">Neutral</text>
                    <text style="font-weight:bold;font-size:3.5rem;">{df[df["label"] == "neutral"].shape[0]}</text>
                </div>
                <div style="display:flex;flex-direction:column;">
                    <text style="font-size:1rem;transform:translateY(15px)">Negative</text>
                    <text style="font-weight:bold;font-size:3.5rem;">{df[df["label"] == "negative"].shape[0]}</text>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown(f"""
                    <div style="display:flex;flex-direction:column;margin-bottom:10px">
                        <text style="font-size:1rem;">Data Preview</text>
                    </div>
                """, unsafe_allow_html=True)
    st.dataframe(df.head(), width=1000)


def word_cloud(df):
    def custom_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
        return random.choice(px.colors.qualitative.Pastel)

    text_data = df['text'].tolist()
    wordcloud = WordCloud(
        background_color='#181C14',
        width=800, height=400,
        color_func=custom_color_func
    ).generate(" ".join(text_data))
    fig, ax = plt.subplots()
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)
    plt.clf()


def word_frequency(df):
    text_data = df['text'].tolist()
    tokens = " ".join(text_data).split()
    tokens = [token.lower() for token in tokens if token.isalpha()]
    word_freq = Counter(tokens)
    word_freq_df = pd.DataFrame(word_freq.items(), columns=['Word', 'Frequency']).sort_values(by='Frequency',
                                                                                              ascending=False)
    top_words = word_freq_df.head(10).sort_values(by='Frequency', ascending=True)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=top_words['Frequency'],
        y=top_words['Word'],
        orientation='h',
        marker_color='#ECDFCC',
    ))
    fig.update_layout(
        title='Words Frequency',
        title_font=dict(size=24),
        xaxis_title='Frequency',
        yaxis_title='Word',
        font=dict(size=14, color='#3C3D37'),
    )
    st.plotly_chart(fig)
    plt.clf()


def clustering(df):
    num_clusters = st.slider("Number of Clusters", 2, 10)
    df_plot = text_cluster(df, num_clusters)
    fig = px.scatter(df_plot, x='PC1', y='PC2', color=df_plot['Cluster'].astype(str),
                     title='Clustering Plot',
                     hover_data=['PC1', 'PC2', 'Cluster'],
                     color_discrete_sequence=px.colors.qualitative.Pastel)

    fig.update_layout(
        plot_bgcolor='#181C14',
        title_font=dict(size=16),
        font=dict(size=16, color='#3C3D37')
    )

    st.plotly_chart(fig)


def sentiment_analysis(df):
    text = st.text_area("Enter your text here", height=100)
    if st.button("Process Sentiment") and text:
        df = text_sentiment(text)
        st.markdown(f"""
                <div style="display:flex;flex-direction:column;justify-content:center;align-items:center">
                    <text style="font-size:1rem;transform:translateY(15px)">Sentiment</text>
                    <text style="font-weight:bold;font-size:3.5rem;">{df["label"][0].capitalize()}</text>
                </div>
        """, unsafe_allow_html=True)


def main():
    if not os.path.exists("data/cleaned-data.csv"):
        st.switch_page("App.py")

    st.set_page_config(initial_sidebar_state="collapsed")
    with open("src/style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    header()
    if st.button("Back", key="back"):
        os.remove("data/cleaned-data.csv")
        os.remove("data/raw-data.csv")
        st.switch_page("App.py")
    tab_menu()


if __name__ == "__main__":
    main()
