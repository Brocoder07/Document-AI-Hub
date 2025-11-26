import streamlit as st

def render_utilities(api):
    st.header("🛠️ Utilities")
    
    tab1, tab2, tab3 = st.tabs(["🔍 Similarity Search", "📝 Summarization", "🔤 Embeddings"])
    
    # --- TAB 1: SEARCH ---
    with tab1:
        st.subheader("Vector Similarity Search")
        query = st.text_input("Enter search query", placeholder="Find documents about...")
        top_k = st.slider("Number of results", 1, 10, 5)
        
        if st.button("Search", type="primary"):
            if query:
                with st.spinner("Searching..."):
                    # The API returns a List[SearchResult] directly
                    res = api.similarity_search(query, st.session_state.access_token, top_k)
                
                # SENIOR ENG FIX: Handle List response vs Error Dict
                if isinstance(res, list):
                    if not res:
                        st.info("No matching documents found.")
                    else:
                        for r in res:
                            # Handle both object/dict access safely
                            score = r.get('score', 0) if isinstance(r, dict) else getattr(r, 'score', 0)
                            text = r.get('text', '') if isinstance(r, dict) else getattr(r, 'text', '')
                            meta = r.get('metadata', {}) if isinstance(r, dict) else getattr(r, 'metadata', {})
                            filename = meta.get('filename', 'Unknown')
                            doc_id = r.get('id', '') if isinstance(r, dict) else getattr(r, 'id', '')

                            with st.expander(f"📄 {filename} (Score: {score:.4f})"):
                                st.markdown("**Content snippet:**")
                                st.text(text[:500] + "..." if len(text) > 500 else text)
                                st.caption(f"ID: {doc_id}")
                                
                elif isinstance(res, dict) and "error" in res:
                    st.error(res["error"])
                else:
                    st.error("Unexpected response format from server.")
            else:
                st.warning("Please enter a query")

    # --- TAB 2: SUMMARIZATION ---
    with tab2:
        st.subheader("Document Summarization")
        summary_text = st.text_area("Enter text to summarize", height=200)
        
        col1, col2 = st.columns(2)
        with col1:
            method = st.selectbox("Method", ["Extractive", "Abstractive"])
        
        if st.button("Summarize"):
            if summary_text:
                with st.spinner("Summarizing..."):
                    res = api.summarize_text(summary_text, st.session_state.access_token, method)
                
                if "summary" in res:
                    st.success("Summary Generated:")
                    st.write(res["summary"])
                else:
                    st.error(res.get("error", "Summarization failed"))
            else:
                st.warning("Please enter text")

    # --- TAB 3: EMBEDDINGS ---
    with tab3:
        st.subheader("Generate Embeddings")
        text_input = st.text_input("Enter text")
        if st.button("Generate Vector"):
            if text_input:
                with st.spinner("Calculating..."):
                    res = api.generate_embeddings(text_input, st.session_state.access_token)
                
                if "embedding" in res:
                    st.json(res["embedding"][:10]) # Show preview of first 10 dims
                    st.caption(f"Vector dimensions: {len(res['embedding'])}")
                else:
                    st.error(res.get("error", "Embedding generation failed"))