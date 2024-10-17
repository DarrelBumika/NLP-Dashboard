from pathlib import Path

import streamlit as st
import pandas as pd

def verify_file(file):
    if file is not None:
        fileType = Path(file.name).suffix
        if fileType == ".csv":
            df = pd.read_csv(file)
        elif fileType == ".xls" or fileType == ".xlsx":
            df = pd.read_excel(file)
        else:
            st.error("Invalid file format. Please upload a CSV, XLS or XLSX file.")
            st.stop()

        if "text" not in df.columns:
            st.error("The uploaded file does not contain a 'text' column.")
        else:
            return df


def header():
    st.markdown("""
            <div style="display:flex;flex-direction:column;align-items:center;margin-bottom:30px">
                <h1 style="font-size:5rem">NLP Dashboard</h1>
                <div style="display:flex;align-items:center;background-color:#3C3D37;padding:20px 20px;border-radius:10px;font-size:1rem;">
                    <text style="text-align: center">This dashboard is designed to simplify text analysis using Natural 
                    Language Processing (NLP) techniques. You can easily upload a CSV, XLS or XLSX file containing text data, and we 
                    will handle the data processing for you.</text>
                </div>
            </div>
        """, unsafe_allow_html=True)


def upload_file():
    file = st.file_uploader("Upload your CSV, XLS or XLSX file", type=["csv", "xls", "xlsx"], accept_multiple_files=False)

    df = verify_file(file)

    if df is not None:
        st.button("Process Data")

        return df
    else:
        st.markdown("""
                <div style="display:flex;flex-direction:column;align-items:center;padding:10px;border:1px solid #697565;border-radius:10px">
                    <text style="text-align: center">Please ensure that your uploaded file includes a column header with the 
                    following format:</text>
                    <ul style="text-align: left; margin:0px">
                        <li>"text" column header for the text data.</li>
                        <li>"label" column header for the sentiment labels (optional).</li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)

def main():
    st.set_page_config(initial_sidebar_state="collapsed")
    with open("src/style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    header()
    df = upload_file()

if __name__ == "__main__":
    main()
