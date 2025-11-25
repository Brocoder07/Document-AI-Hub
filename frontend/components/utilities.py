import streamlit as st
import json

def render_utilities_page(api):
    st.title("🛠️ AI Utilities")
    token = st.session_state.access_token

    tab1, tab2, tab3 = st.tabs(["🔍 Similarity Search", "🧬 Embeddings", "🎨 Formatter"])

    # --- 1. Similarity Search ---
    with tab1:
        st.subheader("Raw Vector Search")
        query = st.text_input("Search Query", "What is the termination clause?")
        top_k = st.slider("Top K Results", 1, 10, 3)
        
        if st.button("Search"):
            with st.spinner("Searching vector DB..."):
                res = api.similarity_search(query, token, top_k)
            
            if "error" in res:
                st.error(res["error"])
            else:
                results = res.get("results", [])
                for r in results:
                    with st.container():
                        st.markdown(f"**Score:** `{r['score']:.4f}` | **ID:** `{r['document_id']}`")
                        # Handle metadata safely
                        meta = r.get("metadata") or r.get("meta") or {}
                        st.caption(f"File: {meta.get('filename', 'Unknown')}")
                        st.text(r.get("text", "")[:300] + "...")
                        st.divider()

    # --- 2. Embeddings ---
    with tab2:
        st.subheader("Generate Vector Embeddings")
        text_to_embed = st.text_area("Enter text to embed", "Hello world")
        if st.button("Generate Vector"):
            with st.spinner("Calculating..."):
                res = api.generate_embeddings(text_to_embed, token)
            
            if "error" in res:
                st.error(res["error"])
            else:
                vec = res.get("embedding", [])
                st.success(f"Generated {len(vec)}-dimensional vector")
                st.code(json.dumps(vec[:10]) + " ... (truncated)", language="json")

    # --- 3. Formatter ---
    with tab3:
        st.subheader("Response Formatter")
        raw_text = st.text_area("Raw Text", "Here is a list: item 1 item 2")
        fmt_type = st.radio("Format To:", ["markdown", "json"])
        
        if st.button("Format"):
            with st.spinner("Formatting..."):
                res = api.format_response(raw_text, token, fmt_type)
            
            if "error" in res:
                st.error(res["error"])
            else:
                st.markdown("### Result:")
                if fmt_type == "json":
                    st.json(res)
                else:
                    st.markdown(res.get("formatted_text", ""))