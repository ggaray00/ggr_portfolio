import streamlit as st

PASSWORD = st.secrets["log_in_pwd"]  # Change this to your desired password

from huggingface_hub import InferenceClient

st.set_page_config(layout="wide")
# client = InferenceClient("dslim/distilbert-NER",
#                          token=st.secrets["all_token"])
#
#
# aa = client.token_classification("Hello, my name is John Doe and I live in New York City.")
#
# print(aa)

# Check if user is logged in
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

def login():
    if st.session_state["password"] == PASSWORD:
        st.session_state.authenticated = True
        st.success("Login successful!")
    else:
        st.error("Incorrect password. Try again.")

def logout():
    st.session_state.authenticated = False
    st.success("Logged out successfully!")

# Display login page if not authenticated
if not st.session_state.authenticated:
    st.title("🔒 Login")
    st.text_input("Password", type="password", key="password")
    st.button("Login", on_click=login)
    st.stop()

# If logged in, show the main content
st.title("🏠 Welcome to the Streamlit Multipage App!")
st.sidebar.button("Logout", on_click=logout)
st.sidebar.success("Select a page above.")



