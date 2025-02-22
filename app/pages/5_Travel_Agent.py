import streamlit as st
import uuid

# from .gg_agent import get_langgraph
from langchain_core.messages import ToolMessage
import random
import time

from travel_agent.main import get_langgraph

if "authenticated" not in st.session_state or not st.session_state.authenticated:
    st.warning("🔒 Please login from the main page to access this page.")
    st.stop()

part_4_graph = get_langgraph(st.secrets["anthropic_api_key"])

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


def process_message(message: str):
    state = {"messages": [("user", message)]}
    # state = part_4_graph.stream(state, config=st.session_state.config)

    try:
        result = part_4_graph.invoke(state, config=st.session_state.config)
        snapshot = part_4_graph.get_state(st.session_state.config)
        if snapshot.next:
            st.session_state.awaiting_approval = snapshot
            return "Please approve or deny the requested action."


        if "messages" in result:
            # print("***********************************")
            # print(result["messages"][-1])
            try:
                rt = result["messages"][-1].content

            except:
                # print(result["messages"][-1][0]['text'])
                rt = result["messages"][-1][0]['text']
            if isinstance(rt, list):
                rt = rt[0]['text']
            # print(f"{rt=}")
            return rt
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
                            tool_call_id=st.session_state.awaiting_approval["messages"][-1].tool_calls[0]["id"],
                            content=f"API call denied by user. Reasoning: '{reason}'. Continue assisting, accounting for the user's input.",
                        )
                    ]
                },
                st.session_state.config,
            )

        st.session_state.awaiting_approval = None

        rt = result["messages"][-1].content if "messages" in result else "Action processed."
        if isinstance(rt, list):
            rt = rt[0]['text']
        return rt
        # return result["messages"][-1].content if "messages" in result else "Action processed."
    except Exception as e:
        st.session_state.awaiting_approval = None
        return f"Error processing approval: {str(e)}"


# UI Layout
st.title("Swiss Airlines Travel Assistant")
init_session_state()



# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if st.session_state.awaiting_approval:
    # st.warning("The assistant needs your approval for the next action.")
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
        print(f"{prompt=}")
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        response = process_message(prompt)
        print(f"{response=}")
        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.write(response)
        if response == "Please approve or deny the requested action.":
            st.rerun()



