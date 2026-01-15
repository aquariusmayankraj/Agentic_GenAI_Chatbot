from logging import PlaceHolder
import streamlit as st

uploaded_file = st.file_uploader("Upload PDF", type="pdf", accept_multiple_files=False)

user_query = st.text_area("Enter your Prompt: ", height=150, placeholder="Ask Anything about the PDF...")

ask_question = st.button("Ask Question")

if ask_question:
    if uploaded_file:
        st.chat_message("user").write(user_query)
        fixed_response = "Hi, This is a fixed response!"
        st.chat_message("AI Lawyer").write(fixed_response)
    else:
        st.error("Please upload a PDF file first!")