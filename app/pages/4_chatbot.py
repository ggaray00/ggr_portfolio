import time
import streamlit as st
import uuid
import sys
import os
from langchain_core.messages import ToolMessage

if "authenticated" not in st.session_state or not st.session_state.authenticated:
    st.warning("🔒 Please login from the main page to access this page.")
    st.stop()

# Ensure project root is in sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from app.travel_agent.graph import part_4_graph


def init_session_state():
    """Initialize Streamlit session state variables."""
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
    """Process user message and return response from the assistant."""
    try:
        state = {"messages": [("user", message)]}
        result = part_4_graph.invoke(state, config=st.session_state.config)
        snapshot = part_4_graph.get_state(st.session_state.config)

        # If approval is required
        if snapshot.next:
            st.session_state.awaiting_approval = snapshot
            content = result["messages"][-1].content
            return content if isinstance(content, str) else content[0].get('text', "No response available")

        # Normal response handling
        if isinstance(result, dict) and "messages" in result:
            content = result["messages"][-1].content
            return content if isinstance(content, str) else content[0].get('text', "No response available")

        return "I couldn't process your request. Please try again."

    except Exception as e:
        return f"An error occurred: {str(e)}"


def handle_approval(approved: bool, reason: str = ""):
    """Handle user approval/denial of an action."""
    if not st.session_state.awaiting_approval:
        return

    try:
        # If approved, invoke next step
        if approved:
            result = part_4_graph.invoke(None, st.session_state.config)
        else:
            # If denied, provide reasoning
            result = part_4_graph.invoke(
                {
                    "messages": [
                        ToolMessage(
                            tool_call_id=st.session_state.awaiting_approval.messages[-1].tool_calls[0]["id"],
                            content=f"API call denied by user. Reason: '{reason}'. Continue assisting the user."
                        )
                    ]
                },
                st.session_state.config
            )

        # Reset approval state
        st.session_state.awaiting_approval = None

        if isinstance(result, dict) and "messages" in result:
            content = result["messages"][-1].content
            return content if isinstance(content, str) else content[0].get('text', "Action processed")

        return "Action processed."

    except Exception as e:
        st.session_state.awaiting_approval = None
        return f"Error processing approval: {str(e)}"


# UI Layout
st.title("🛫 Travel Assistant")
st.caption("Your personal travel assistant for flight, car rental, and hotel queries.")
init_session_state()

with st.expander("ℹ️ About this app"):
    st.write(
        """
        This is an AI-powered travel assistant bot built with LangChain and LangGraph.
        It connects different assistants with tools such as:
        - A RAG (Retrieval-Augmented Generation) pipeline for airline policies.
        - A database tool for fetching flight details.

        Ask about your flight, car rental options, hotel booking, or weekend getaways!
        """
    )

# Predefined questions
example_questions = [
    "At what time is my flight?",
    "What car rental options do I have in Basel?",
    "Could you book a hotel?",
    "Can I change my flight?",
    "Can you suggest a weekend getaway near me?",
]

selected_question = st.selectbox(
    "💬 Example questions (or ask your own):",
    ["Select a question..."] + example_questions,
    index=(["Select a question..."] + example_questions).index(st.session_state.selected_question),
)

# Process pre-selected question
if selected_question != "Select a question..." and selected_question != st.session_state.selected_question:
    st.session_state.selected_question = selected_question  # Store selection
    st.session_state.messages.append({"role": "user", "content": selected_question})

    with st.spinner("Thinking... 💭"):
        response = process_message(selected_question)

    st.session_state.messages.append({"role": "assistant", "content": response})

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Approval handling
if st.session_state.awaiting_approval:
    with st.form("approval_form"):
        st.write("🔍 Action pending approval:")
        col1, col2 = st.columns(2)
        with col1:
            approve = st.form_submit_button("✅ Approve")
        with col2:
            deny = st.form_submit_button("❌ Deny")

        if deny:
            reason = st.text_area("Please explain why you're denying this action:")
            submit_denial = st.form_submit_button("Submit Denial")

        if approve:
            response = handle_approval(True)
            if response:
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.rerun()

        if deny and submit_denial:
            response = handle_approval(False, reason)
            if response:
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.rerun()

# Chat input
if not st.session_state.awaiting_approval:
    if prompt := st.chat_input("💡 How can I help you today?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        with st.spinner("Thinking... 🤔"):
            response = process_message(prompt)

        with st.chat_message("assistant"):
            placeholder = st.empty()
            full_response = ""
            # Simulate typing effect
            for line in response.splitlines(keepends=True):
                full_response += line
                time.sleep(0.2)
                placeholder.markdown(full_response + "▌")
            placeholder.markdown(full_response)

        st.session_state.messages.append({"role": "assistant", "content": full_response})
