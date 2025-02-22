import streamlit as st
import pandas as pd
import pdfplumber
from annotated_text import annotated_text
from huggingface_hub import InferenceClient
from st_ner_annotate import st_ner_annotate
from st_copy_to_clipboard import st_copy_to_clipboard
from annotated_text import annotated_text

annotated_text(
    "This ",
    ("is", "verb"),
    " some ",
    ("annotated", "adj"),
    ("text", "noun"),
    " for those of ",
    ("you", "pronoun"),
    " who ",
    ("like", "verb"),
    " this sort of ",
    ("thing", "noun"),
    "."
)



if "authenticated" not in st.session_state or not st.session_state.authenticated:
    st.warning("🔒 Please login from the main page to access this page.")
    st.stop()

client = InferenceClient("dslim/distilbert-NER", token=st.secrets["all_token"])

def viz_entities(text):
    entities = client.token_classification(text)
    for entity in entities:
        annotated_text(
            text[:entity["start"]],
            (text[entity["start"]:entity["end"]], entity["entity"]),
            text[entity["end"]:]
        )


st.title("🔎 Entity Extraction Tool")

uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
if uploaded_file:
    with pdfplumber.open(uploaded_file) as pdf:
        text = ''
        for page in pdf.pages:
            text += page.extract_text()

    st.write(text)
    viz_entities(text)

st_copy_to_clipboard("Copy this to clipboard")

st.title("Named entity recognition demo")
text = """Manhattan traces its origins to a trading post founded by colonists 
    from the Dutch Republic in 1624 on Lower Manhattan; the post was named New 
    Amsterdam in 1626. Manhattan is historically documented to have been purchased 
    by Dutch colonists from Native Americans in 1626 for 60 guilders, which equals 
    roughly $1059 in current terms. The territory and its surroundings came under 
    English control in 1664 and were renamed New York after King Charles II of 
    England granted the lands to his brother, the Duke of York. New York, based 
    in present-day Manhattan, served as the capital of the United States from 1785 
    until 1790. The Statue of Liberty greeted millions of immigrants as they came 
    to America by ship in the late 19th century and is a world symbol of the United 
    States and its ideals of liberty and peace. Manhattan became a borough during 
    the consolidation of New York City in 1898."""

entity_labels = ["LOC", "PERSON", "ORG", "MISC"]
ents = []


current_entity_type = st.selectbox("Mark for Entity Type", entity_labels)
entities = st_ner_annotate(current_entity_type, text, ents)
st.json(entities)
st.json(ents)
