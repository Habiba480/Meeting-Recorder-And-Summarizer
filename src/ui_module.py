import streamlit as st

def set_page_config_and_style():
    st.set_page_config(page_title="AI Meeting Summarizer", layout="wide")
    st.markdown("""
        <style>
        .block-container { padding-top: 1rem; }
        .sidebar .sidebar-content { background-color: #a2d2ff; }
        </style>
    """, unsafe_allow_html=True)

def setup_sidebar():
    st.sidebar.title(" Meeting Chats")
    if "chat_titles" not in st.session_state:
        st.session_state.chat_titles = []
