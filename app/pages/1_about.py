import streamlit as st

if "authenticated" not in st.session_state or not st.session_state.authenticated:
    st.warning("🔒 Please login from the main page to access this page.")
    st.stop()

st.write("DB username:", st.secrets["db_username"])

st.title("📄 About")
st.write("This is the About page for the multipage Streamlit app.")
st.write("Use the sidebar to navigate through different pages.")
