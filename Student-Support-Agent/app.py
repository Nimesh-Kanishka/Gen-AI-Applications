import streamlit as st
from support_agent import build_state_graph, get_response

def main():
    # Decrease the vertical padding for horizontal lines (hr) after each message
    st.markdown("""
        <style>
            .stMarkdown hr {
                margin: 2px 0px;
            }
        </style>
    """, unsafe_allow_html=True)
    
    if "agent" not in st.session_state:
        st.session_state.agent = build_state_graph()

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.title("Student Support Agent")

    message = st.chat_input("Ask your questions here...")

    if message:
        st.session_state.chat_history.append(get_response(state_graph=st.session_state.agent,
                                                          message=message,
                                                          chat_history=st.session_state.chat_history[-5:]))

    for msg in st.session_state.chat_history:
        st.write(f"**You:** {msg['message']}")

        st.write("---")

        if msg["subject"] != "general":
            st.write(f"*The system has detected this question is related to {msg['subject']}. {msg['subject'].capitalize()} agent will be used to generate the reply.*")
        st.write(f"**Agent:** {msg['response']}")
        
        st.write("---")

if __name__ == "__main__":
    main()