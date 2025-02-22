import streamlit as st
import random
import time

if "authenticated" not in st.session_state or not st.session_state.authenticated:
    st.warning("🔒 Please login from the main page to access this page.")
    st.stop()

st.write("Streamlit loves LLMs! 🤖 [Build your own chat app](https://docs.streamlit.io/develop/tutorials/llms/build-conversational-apps) in minutes, then make it powerful by adding images, dataframes, or even input widgets to the chat.")

st.caption("Note that this demo app isn't actually connected to any LLMs. Those are expensive ;)")


# Function to simulate assistant response
def get_assistant_response(prompt):
    responses = [
        "Hello there! How can I assist you today?",
        "Hi, human! Is there anything I can help you with?",
        "Do you need help?",
    ]
    return random.choice(responses)

def handle_input(user_input):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    assistant_reply = get_assistant_response(user_input)
    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""
        for chunk in assistant_reply.split():
            full_response += chunk + " "
            time.sleep(0.05)
            placeholder.markdown(full_response + "▌")
        placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})


if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Let's start chatting! 👇"}]

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


user_input = st.chat_input("Enter a message...")
if user_input:
    handle_input(user_input)







