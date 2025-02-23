import time
import streamlit as st
import uuid
import sys
import os
from langchain_core.messages import ToolMessage

# from app.travel_agent.graph import part_4_graph


project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from app.travel_agent.graph import part_4_graph


def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())
    if "awaiting_approval" not in st.session_state:
        st.session_state.awaiting_approval = None
    if "config" not in st.session_state:
        st.session_state.config = {
            "configurable": {
                "passenger_id": "3442 587242",
                "thread_id": st.session_state.thread_id,
            }
        }
    if "selected_question" not in st.session_state:
        st.session_state.selected_question = "Select a question..."


def process_message(message: str):
    try:
        state = {"messages": [("user", message)]}
        result = part_4_graph.invoke(state, config=st.session_state.config)
        snapshot = part_4_graph.get_state(st.session_state.config)
        if snapshot.next:
            st.session_state.awaiting_approval = snapshot
            content = result["messages"][-1].content
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Approve"):
                    response = handle_approval(True)
                    if response:
                        st.session_state.messages.append({"role": "assistant", "content": response})
                        st.rerun()
            with col2:
                if st.button("Deny"):
                    reason = st.text_input("Please explain why you're denying this action:")
                    if st.button("Submit Denial"):
                        response = handle_approval(False, reason)
                        if response:
                            st.session_state.messages.append({"role": "assistant", "content": response})
                            st.rerun()

            if isinstance(content, list):
                return content[0].get('text',
                                      "No response available") + "\n\nPlease approve or deny the requested action."
            return content + "\n\nPlease approve or deny the requested action."

            # return "Please approve or deny the requested action."

        if isinstance(result, dict) and "messages" in result:
            content = result["messages"][-1].content
            if isinstance(content, list):
                return content[0].get('text', "No response available")
            return content

        return "I couldn't process your request. Please try again."

    except Exception as e:
        return f"An error occurred: {str(e)}"


def handle_approval(approved: bool, reason: str = ""):
    if not st.session_state.awaiting_approval:
        return

    try:
        if approved:
            result = part_4_graph.invoke(None, st.session_state.config)
        else:
            result = part_4_graph.invoke(
                {
                    "messages": [
                        ToolMessage(
                            tool_call_id=st.session_state.awaiting_approval.messages[-1].tool_calls[0]["id"],
                            content=f"API call denied by user. Reasoning: '{reason}'. Continue assisting, accounting for the user's input."
                        )
                    ]
                },
                st.session_state.config
            )

        st.session_state.awaiting_approval = None

        if isinstance(result, dict) and "messages" in result:
            content = result["messages"][-1].content
            if isinstance(content, list):
                return content[0].get('text', "Action processed")
            return content

        return "Action processed"

    except Exception as e:
        st.session_state.awaiting_approval = None
        return f"Error processing approval: {str(e)}"


# UI Layout
st.subheader("Travel Assistant")
st.caption("Your personal travel assistant for flight, car rental, and hotel queries.")
st.caption("In this example the user has a flight booked form Paris to Basel.")
init_session_state()

with st.expander("About this app"):
    st.write(
        """
        This is an AI-powered travel assistant bot built with LangChain and LangGraph.
        It connects different assistants with tools such as:
        - A RAG (Retrieval-Augmented Generation) pipeline for airline policies.
        - A database tool for fetching flight details.

        Ask about your flight, car rental options, hotel booking, or weekend getaways!
        """
    )

example_questions = [
    "At what time is my flight?",
    "What car rental options do I have in Basel?",
    "Could you book a hotel?",
    "Can I change my flight?",
    "Can you suggest a weekend getaway near me?",
]

selected_question = st.selectbox(
    "Example questions (or ask your own):",
    ["Select a question..."] + example_questions,
    index=(["Select a question..."] + example_questions).index(st.session_state.selected_question),
)

if selected_question != "Select a question..." and selected_question != st.session_state.selected_question:
    st.session_state.selected_question = selected_question  # Store selection
    st.session_state.messages.append({"role": "user", "content": selected_question})

    with st.spinner("Thinking... "):
        response = process_message(selected_question)

    st.session_state.messages.append({"role": "assistant", "content": response})


# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Handle approval UI if needed
if st.session_state.awaiting_approval:
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Approve"):
            response = handle_approval(True)
            if response:
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.rerun()
    with col2:
        if st.button("Deny"):
            reason = st.text_input("Please explain why you're denying this action:")
            if st.button("Submit Denial"):
                response = handle_approval(False, reason)
                if response:
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    st.rerun()

# Chat input
if not st.session_state.awaiting_approval:
    if prompt := st.chat_input("How can I help you today?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        with st.spinner("Thinking... "):
            response = process_message(prompt)

        with st.chat_message("assistant"):
            placeholder = st.empty()
            full_response = ""
            # Iterate over each line in the response to preserve newlines.
            for line in response.splitlines(keepends=True):
                full_response += line
                time.sleep(0.2)
                placeholder.markdown(full_response + "▌")
            # Final display without duplicating the message.
            placeholder.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})

