import random
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud


def main():
    st.set_page_config(initial_sidebar_state="collapsed")

    df = pd.read_csv("tweets-data/sentiment-data.csv")

    st.markdown("""
                <div style="display:flex;flex-direction:column;align-items:center;margin-bottom:30px">
                    <h1 style="font-size:4rem">NLP Dashboard</h1>
                    <div style="display:flex;width:100%;background-color:#b17457;padding:0px 20px;border-radius:10px;justify-content:center">
                        <text style="color:white;font-size:25px">Twitter Analysis of "Gibran"</text>
                    </div>
                </div>
            """, unsafe_allow_html=True)

    st.markdown("""
        <div style="display:flex;justify-content:center;align-items:center;gap:10px;margin-bottom:30px">
            <div style="height:2px;width:100%;background-color:#d8d2c2"></div>
            <text style="font-size:1.2rem;font-weight:bold;color:#d8d2c2">Overview</text>
            <div style="height:2px;width:100%;background-color:#d8d2c2"></div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("""
            <div style="display:flex;flex-direction:column;border:1px;align-items:center">
                <text style="font-size:1.2rem;font-weight:bold;color:#b17457">Keywords</text>
                <text style="font-size:1.5rem;font-weight:bolder">"Gibran since:2024-09-01 until:2024-09-30 lang:id"</text>
                <text style="font-size:1.5rem;font-weight:bolder">"Gibran since:2024-10-01 lang:id"</text>
            </div>
        """, unsafe_allow_html=True)

    st.markdown(f"""
        <div style="display:flex;flex-direction:column;align-items:center">
            <text style="font-size:1.2rem;font-weight:bold;color:#b17457;transform:translateY(30px)">Data Found</text>
            <text style="font-size:5rem;font-weight:bolder">{len(df)}</text>
        </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        df_count = df["sentiment"].value_counts()
        df_count = pd.DataFrame(df_count)
        df_count.reset_index(inplace=True)
        df_count.columns = ["sentiment", "count"]

        fig = px.pie(df_count, values='count', names='sentiment', hole=0.5,
                     color_discrete_sequence=['#b17457', '#4a4947', '#d8d2c2'])

        fig.update_traces(
            textinfo='percent',
            textfont_size=16,
            textfont_color='white',
            textposition='inside'
        )

        fig.update_layout(
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.2,
                xanchor="center",
                x=0.5
            )
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown(f"""
            <div>
                <div style="height:450px;display:flex;flex-direction:column;justify-content:space-between;align-items:center">
                    <div style="display:flex;flex-direction:column;align-items:center">
                        <text style="font-size:3rem;font-weight:bolder">{len(df[df["sentiment"] == "positive"])}</text>
                        <text style="font-size:1rem;font-weight:bold;color:#b17457;transform:translateY(-15px)">Positive</text>
                    </div>
                    <div style="height:2px;width:150px;background-color:#d8d2c2"></div>
                    <div style="display:flex;flex-direction:column;align-items:center">
                        <text style="font-size:3rem;font-weight:bolder">{len(df[df["sentiment"] == "negative"])}</text>
                        <text style="font-size:1rem;font-weight:bold;color:#b17457;transform:translateY(-15px)">Negative</text>
                    </div>
                    <div style="height:2px;width:150px;background-color:#d8d2c2"></div>
                    <div style="display:flex;flex-direction:column;align-items:center">
                        <text style="font-size:3rem;font-weight:bolder">{len(df[df["sentiment"] == "neutral"])}</text>
                        <text style="font-size:1rem;font-weight:bold;color:#b17457;transform:translateY(-15px)">Neutral</text>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("""
            <div style="display:flex;justify-content:center;align-items:center;gap:10px;margin-bottom:30px">
                <div style="height:2px;width:100%;background-color:#d8d2c2"></div>
                <text style="text-align:center;font-size:1.2rem;font-weight:bold;color:#d8d2c2;width:auto">Words Frequency</text>
                <div style="height:2px;width:100%;background-color:#d8d2c2"></div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("""
        <div style="display:flex;flex-direction:column;border:1px;align-items:center">
            <text style="font-size:1.2rem;font-weight:bold;color:#b17457">Word Cloud</text>
        </div>
    """, unsafe_allow_html=True)

    def custom_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
        colors = ['#b17457', '#4a4947', '#d8d2c2']
        return random.choice(colors)

    words = " ".join(df["Tweet"])
    wordcloud = WordCloud(width=800, height=400, background_color="#faf7f0", color_func=custom_color_func).generate(words)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')

    st.pyplot(fig)

    st.markdown("""
            <div style="display:flex;flex-direction:column;border:1px;align-items:center;margin-top:20px">
                <text style="font-size:1.2rem;font-weight:bold;color:#b17457">Top 10 Words</text>
            </div>
        """, unsafe_allow_html=True)

    word_freq = wordcloud.words_
    top_words = dict(sorted(word_freq.items(), key=lambda item: item[1], reverse=True)[:10])

    fig, ax = plt.subplots(figsize=(10, 5))
    top_words = dict(sorted(word_freq.items(), key=lambda item: item[1], reverse=True)[:10])

    top_words_df = pd.DataFrame(top_words.items(), columns=['Kata', 'Frekuensi'])

    fig = px.bar(top_words_df, x='Kata', y='Frekuensi',
                 color='Frekuensi',
                 color_continuous_scale=['#d8d2c2', '#b17457', '#4a4947'])

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
                <div style="display:flex;justify-content:center;align-items:center;gap:10px;margin-bottom:30px">
                    <div style="height:2px;width:100%;background-color:#d8d2c2"></div>
                    <text style="text-align:center;font-size:1.2rem;font-weight:bold;color:#d8d2c2;width:auto">Clustering</text>
                    <div style="height:2px;width:100%;background-color:#d8d2c2"></div>
                </div>
            """, unsafe_allow_html=True)

    colors = ['#b17457', '#4a4947', '#d8d2c2']

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df['Tweet']).toarray()

    true_k = 3
    model = KMeans(n_clusters=true_k, init='k-means++', max_iter=1000, n_init=true_k)
    model.fit(X)

    pca = PCA(n_components=2, random_state=0)
    reduced_features = pca.fit_transform(X)

    reduced_cluster_centers = pca.transform(model.cluster_centers_)

    plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=[colors[i] for i in model.predict(X)])
    plt.scatter(reduced_cluster_centers[:, 0], reduced_cluster_centers[:, 1], marker='x', s=150, c='b')

    st.pyplot(plt)

if __name__ == "__main__":
    main()
