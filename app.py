# app.py
import streamlit as st
from pipeline.chatbot import ask_chatbot

# Set up the page
st.set_page_config(page_title="Vidensbasen RAG Chatbot", page_icon="⚖️")
st.title("⚖️ Anklagemyndigheden Chatbot")
st.markdown("Spørg om danske juridiske retningslinjer (Ask about Danish legal guidelines).")

# --- NEW: Model Selection in the Sidebar ---
with st.sidebar:
    st.header("⚙️ Indstillinger (Settings)")
    selected_model = st.selectbox(
        "Vælg en sprogmodel (Choose a model):",
        options=["gpt-5-nano", "gpt-5-mini"],
        index=0, # Defaults to the first option (nano)
        help="Nano is faster and strictly follows instructions. Mini might offer more elaborate answers."
    )

# Initialize chat history in Streamlit's session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Hvad vil du gerne vide?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        with st.spinner("Søger i Vidensbasen... (Searching...)"):
            # Call the chatbot function we built in Phase 4
            response = ask_chatbot(prompt)
            st.markdown(response)
            
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})