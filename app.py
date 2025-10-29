"""
Clinical Trial RAG Chatbot - Streamlit Web UI
"""
import streamlit as st
import json
from chatbot import RAGChatbot
import time


# Page configuration
st.set_page_config(
    page_title="Clinical Trial AI Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styles
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .source-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #1f77b4;
    }
    .time-stat {
        background-color: #e8f4f8;
        padding: 0.5rem;
        border-radius: 0.3rem;
        margin: 0.3rem 0;
    }
    .answer-box {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border: 1px solid #e0e0e0;
        margin: 1rem 0;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_chatbot():
    """Load chatbot (cached)"""
    try:
        return RAGChatbot()
    except Exception as e:
        st.error(f"Failed to initialize chatbot: {e}")
        st.info("Please make sure you have run indexer.py to create the index")
        return None


def display_sources(sources):
    """Display source information"""
    if not sources:
        st.warning("No relevant clinical trials found")
        return
    
    st.markdown("### 📚 Sources")
    for i, source in enumerate(sources, 1):
        with st.container():
            st.markdown(f"""
            <div class="source-card">
                <strong>Source {i}: {source['NCTId']}</strong><br>
                <em>{source['Title']}</em><br>
                <small>Relevance: {source['Relevance']}</small>
            </div>
            """, unsafe_allow_html=True)


def display_timing(timing):
    """Display timing statistics"""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="time-stat">
            <strong>Retrieval Time</strong><br>
            {timing['retrieval_time']}
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="time-stat">
            <strong>API Call Time</strong><br>
            {timing['api_time']}
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="time-stat">
            <strong>Total Time</strong><br>
            {timing['total_time']}
        </div>
        """, unsafe_allow_html=True)


def main():
    """Main function"""
    
    # Page title
    st.markdown('<div class="main-header">🏥 Clinical Trial AI Assistant</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Clinical Trial Information Retrieval System Based on RAG Technology</div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 📊 System Information")
        st.info("""
        This system uses RAG (Retrieval-Augmented Generation) technology:
        
        - 🔍 Fast retrieval of relevant clinical trials
        - 💬 Answer questions about trials
        - 📝 Provide accurate source citations
        - ⚡ Low latency response (<2s)
        """)
        
        st.markdown("## 🎯 User Guide")
        st.markdown("""
        1. Enter your question in the input box below
        2. Click "Send" or press Enter
        3. View AI assistant's answer
        4. Check source accuracy
        """)
        
        st.markdown("## 💡 Example Questions")
        example_queries = [
            "Are there any clinical trials for PTSD?",
            "Which trials accept healthy volunteers?",
            "Are there any studies for adolescents?",
            "What cancer trials are recruiting?",
            "Which trials are conducted in California?"
        ]
        
        for query in example_queries:
            if st.button(query, key=query):
                st.session_state['query_input'] = query
    
    # Initialize session state
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []
    if 'query_input' not in st.session_state:
        st.session_state['query_input'] = ""
    
    # Load chatbot
    chatbot = load_chatbot()
    
    if chatbot is None:
        st.error("❌ Cannot load chatbot. Please run `python indexer.py` to create index first.")
        return
    
    # Main content area
    st.markdown("---")
    
    # Input area
    query = st.text_input(
        "💬 Please enter your question:",
        value=st.session_state.get('query_input', ''),
        placeholder="e.g., Are there any clinical trials for diabetes?",
        key="main_input"
    )
    
    col1, col2, col3 = st.columns([1, 1, 4])
    with col1:
        send_button = st.button("🚀 Send", type="primary")
    with col2:
        clear_button = st.button("🗑️ Clear History")
    
    # Clear history
    if clear_button:
        st.session_state['chat_history'] = []
        st.session_state['query_input'] = ""
        st.rerun()
    
    # Process query
    if send_button and query:
        with st.spinner("🔍 Retrieving relevant information..."):
            # Call chatbot
            result = chatbot.chat(query)
            
            # Add to history
            st.session_state['chat_history'].append({
                'query': query,
                'result': result
            })
            
            # Clear input
            st.session_state['query_input'] = ""
    
    # Display conversation history
    if st.session_state['chat_history']:
        st.markdown("---")
        st.markdown("## 💬 Conversation History")
        
        # Display in reverse order (newest first)
        for i, item in enumerate(reversed(st.session_state['chat_history'])):
            query_text = item['query']
            result = item['result']
            
            # Question
            st.markdown(f"### 🙋 Question {len(st.session_state['chat_history']) - i}")
            st.markdown(f"**{query_text}**")
            
            # Answer
            st.markdown("### 🤖 Answer")
            st.markdown(f"""
            <div class="answer-box">
                {result['answer']}
            </div>
            """, unsafe_allow_html=True)
            
            # Sources and timing
            col1, col2 = st.columns([2, 1])
            
            with col1:
                display_sources(result['sources'])
            
            with col2:
                st.markdown("### ⏱️ Performance Metrics")
                display_timing(result['timing'])
            
            st.markdown("---")
    
    # Bottom statistics
    if chatbot and st.session_state['chat_history']:
        st.markdown("## 📈 Overall Statistics")
        stats = chatbot.get_stats()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Queries", stats['total_queries'])
        with col2:
            st.metric("Avg Retrieval Time", stats['avg_retrieval_time'])
        with col3:
            st.metric("Avg API Time", stats['avg_api_time'])
        with col4:
            st.metric("Avg Total Time", stats['avg_total_time'])
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9rem;">
        <p>⚕️ Clinical Trial AI Assistant</p>
        <p>Based on Retrieval-Augmented Generation (RAG) Technology</p>
        <p>⚠️ This system is for reference only. Please consult professional medical personnel for specific information.</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
