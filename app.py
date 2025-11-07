"""
Minimal Streamlit UI for testing Cognito-Droid RAG Chatbot
"""

import streamlit as st
from chatbot import CognitoDroidChatbot

# Page config
st.set_page_config(
    page_title="Cognito-Droid Chatbot",
    page_icon="🤖",
    layout="wide"
)

# Initialize chatbot (with caching)
@st.cache_resource
def load_chatbot():
    return CognitoDroidChatbot()

# Session state for messages
if "messages" not in st.session_state:
    st.session_state.messages = []

if "mode" not in st.session_state:
    st.session_state.mode = "learn"

if "custom_instructions" not in st.session_state:
    st.session_state.custom_instructions = ""

# Load chatbot
try:
    chatbot = load_chatbot()
    chatbot.set_mode(st.session_state.mode)
    
    # Sidebar - Mode Selection
    st.sidebar.title("🎯 Mode Selection")
    
    mode = st.sidebar.selectbox(
        "Choose Mode:",
        ["learn", "hint", "quiz", "eli5", "custom"],
        index=["learn", "hint", "quiz", "eli5", "custom"].index(st.session_state.mode)
    )
    
    # Update mode if changed
    if mode != st.session_state.mode:
        st.session_state.mode = mode
        chatbot.set_mode(mode)
    
    # Custom mode instructions
    if mode == "custom":
        st.sidebar.subheader("Custom Instructions")
        custom_input = st.sidebar.text_area(
            "Define how the chatbot should behave:",
            value=st.session_state.custom_instructions,
            placeholder="e.g., Always respond with bullet points and include code examples"
        )
        
        if st.sidebar.button("Apply Custom Mode"):
            st.session_state.custom_instructions = custom_input
            chatbot.set_custom_mode(custom_input)
            st.sidebar.success("Custom mode applied!")
    
    # Mode descriptions
    mode_descriptions = {
        "learn": "📚 Explains concepts with examples and clear breakdowns",
        "hint": "💡 Provides Socratic hints without giving direct answers",
        "quiz": "📝 Generates practice questions and evaluates answers",
        "eli5": "👶 Explains concepts in simple, easy-to-understand language",
        "custom": "⚙️ Follows your custom instructions"
    }
    
    st.sidebar.info(mode_descriptions[mode])
    
    # Clear conversation button
    if st.sidebar.button("🗑️ Clear Conversation"):
        st.session_state.messages = []
        chatbot.clear_memory()
        st.sidebar.success("Conversation cleared!")
    
    # Main chat interface
    st.title("🤖 Cognito-Droid RAG Chatbot")
    st.caption(f"Current Mode: **{mode.upper()}**")
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your study materials..."):
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = chatbot.chat(prompt)
                st.markdown(response)
        
        # Add assistant response to chat
        st.session_state.messages.append({"role": "assistant", "content": response})

except FileNotFoundError:
    st.error("❌ Vector store not found! Please ingest documents first.")
    st.info("Run: `python main.py` and select option 1 to ingest documents.")
except Exception as e:
    st.error(f"❌ Error: {e}")
    import traceback
    st.code(traceback.format_exc())

# Footer
st.sidebar.markdown("---")
st.sidebar.caption("Cognito-Droid v1.0 | Arm AI Hackathon")

