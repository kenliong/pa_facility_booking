from utils import *
import streamlit as st

st.title("💬 PA facility booking concierge")

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():

    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    response = respond_to_user_input(prompt)

    st.session_state.messages.append({"role": "assistant", "content": response.content})
    st.chat_message("assistant").write(response.content)