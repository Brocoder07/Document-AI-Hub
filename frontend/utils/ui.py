import streamlit as st
import time

def load_custom_css():
    """Injects custom CSS for better message styling and source cards."""
    st.markdown("""
        <style>
        /* Chat Message Styling */
        .stChatMessage {
            background-color: transparent;
            border-radius: 10px;
            padding: 10px;
        }
        
        /* Source Card Container */
        .source-card {
            background-color: #f0f2f6;
            border: 1px solid #d1d5db;
            border-radius: 8px;
            padding: 10px;
            margin: 5px;
            font-size: 0.85rem;
            height: 100%;
            transition: transform 0.2s;
        }
        .source-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .source-card h4 {
            margin: 0 0 5px 0;
            font-size: 0.95rem;
            color: #1f2937;
        }
        .source-score {
            font-size: 0.75rem;
            color: #059669; /* Green color for high confidence */
            font-weight: bold;
        }
        
        /* Citation Badge in text */
        .citation-badge {
            background-color: #e5e7eb;
            color: #374151;
            padding: 2px 6px;
            border-radius: 4px;
            font-size: 0.8em;
            font-weight: 500;
            margin-left: 4px;
            border: 1px solid #d1d5db;
        }
        </style>
    """, unsafe_allow_html=True)

def stream_text(text):
    """Generator function to simulate typing effect."""
    for word in text.split(" "):
        yield word + " "
        time.sleep(0.02)

def format_citation_in_text(text):
    """
    Optional: If your backend returns [DOC ID], replacing it with a cleaner [Source X].
    (This requires regex if you want to dynamically map IDs to numbers, 
     but here is a simple pass-through or cleanup logic).
    """
    # For now, we assume the text is Markdown. 
    # You can add regex here later to turn [DOC...] into <span class='citation-badge'>[1]</span>
    return text