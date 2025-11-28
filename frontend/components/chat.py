import streamlit as st
from utils.ui import load_custom_css
import re

def render_chat_page(api):
    # 1. Load Styles
    load_custom_css()
    
    st.title("💬 Chat with Documents")
    token = st.session_state.access_token

    # Initialize menu state if not exists
    if "open_menus" not in st.session_state:
        st.session_state.open_menus = set()
    if "pending_renames" not in st.session_state:
        st.session_state.pending_renames = {}
    if "confirm_deletes" not in st.session_state:
        st.session_state.confirm_deletes = set()
    if "refresh_sessions" not in st.session_state:
        st.session_state.refresh_sessions = False

    # --- SIDEBAR: HISTORY & CONFIG ---
    with st.sidebar:
        # New Chat Button
        if st.button("➕ New Chat", use_container_width=True, type="primary"):
            st.session_state.chat_session_id = None
            st.session_state.chat_history = []
            st.session_state.open_menus = set()  # Close all menus
            st.session_state.refresh_sessions = True  # Force refresh
            st.rerun()
            
        st.divider()
        
        # History List
        st.subheader("History")
        
        # Check if we need to fetch fresh data or use cache
        if st.session_state.refresh_sessions or "cached_sessions" not in st.session_state:
            with st.spinner("Refreshing chats..."):
                sessions = api.get_chat_sessions(token)
                st.session_state.cached_sessions = sessions
                st.session_state.refresh_sessions = False
        else:
            sessions = st.session_state.cached_sessions
        
        if not sessions:
            st.caption("No previous chats found.")
            
        for s in sessions:
            render_chat_session_item(api, token, s)
            
        st.divider()
        st.header("⚙️ Settings")
        
        # File Context Logic
        try:
            files = api.get_user_files(token)
            if files:
                file_options = {f['filename']: f['file_id'] for f in files}
                file_options["All Documents"] = None
            else:
                file_options = {"All Documents": None}
        except: 
            file_options = {"All Documents": None}
        
        selected_name = st.selectbox("Context", list(file_options.keys()))
        selected_file_id = file_options[selected_name]

    # --- MAIN CHAT AREA ---
    if not st.session_state.chat_history:
        st.info("👋 Start a new conversation or select one from the sidebar!")

    render_chat_messages(api, token)

    # --- INPUT HANDLING ---
    if prompt := st.chat_input("Ask a question about your documents..."):
        handle_user_input(api, token, prompt, selected_file_id, sessions)

def render_chat_session_item(api, token, session):
    """
    Renders a single chat session item with improved layout.
    """
    is_active = (session['id'] == st.session_state.chat_session_id)
    icon = "🟢" if is_active else "💬"
    title = session.get('title') or "New Chat"
    session_id = session['id']
    
    menu_open = session_id in st.session_state.open_menus
    menu_button_key = f"menu_btn_{session_id}"

    # --- ROW LAYOUT: Title + Trigger Button ---
    col1, col2 = st.columns([5, 1])
    
    with col1:
        button_label = f"{icon} {title[:22]}{'...' if len(title) > 22 else ''}"
        if st.button(button_label, key=f"session_{session_id}", use_container_width=True):
            st.session_state.chat_session_id = session_id
            st.session_state.open_menus = set()
            msgs = api.get_session_messages(session_id, token)
            st.session_state.chat_history = [
                {
                    "role": m["role"], 
                    "content": m["content"],
                    "retrieved": m.get("retrieved_docs", [])
                } 
                for m in msgs
            ]
            st.rerun()
    
    with col2:
        if st.button("⚙️", key=menu_button_key, help="Options"):
            if menu_open:
                st.session_state.open_menus.discard(session_id)
            else:
                st.session_state.open_menus.add(session_id)
            st.rerun()

    # --- MENU CONTENT ---
    if menu_open:
        with st.container(border=True):
            
            # --- RENAME SECTION ---
            st.caption("✏️ Rename Chat")
            current_rename = st.session_state.pending_renames.get(session_id, title)
            
            new_title = st.text_input(
                "New Title",
                value=current_rename,
                key=f"rename_input_{session_id}",
                label_visibility="collapsed",
                placeholder="Enter new chat title..."
            )
            
            st.session_state.pending_renames[session_id] = new_title
            
            r_col1, r_col2 = st.columns(2)
            with r_col1:
                if st.button("🔄 Rename", key=f"rename_{session_id}", use_container_width=True, type="primary"):
                    if new_title and new_title != title:
                        with st.spinner("Renaming..."):
                            result = api.update_chat_title(session_id, new_title, token)
                        if "error" not in result:
                            st.success("✅ Chat renamed!")
                            st.session_state.refresh_sessions = True
                            st.session_state.open_menus.discard(session_id)
                            st.session_state.pending_renames.pop(session_id, None)
                            st.rerun()
                        else:
                            st.error(f"❌ Error: {result.get('error')}")
                    elif not new_title:
                        st.error("❌ Title cannot be empty")

            with r_col2:
                if st.button("✖️ Cancel", key=f"cancel_rename_{session_id}", use_container_width=True):
                    st.session_state.open_menus.discard(session_id)
                    st.session_state.pending_renames.pop(session_id, None)
                    st.rerun()
            
            st.divider()
            
            # --- DELETE SECTION ---
            delete_confirmed = session_id in st.session_state.confirm_deletes
            
            if not delete_confirmed:
                if st.button("🗑️ Delete Chat", key=f"delete_init_{session_id}", use_container_width=True):
                    st.session_state.confirm_deletes.add(session_id)
                    st.rerun()
            else:
                st.error("⚠️ Delete this chat?")
                d_col1, d_col2 = st.columns(2)
                with d_col1:
                    if st.button("✅ Yes", key=f"delete_confirm_{session_id}", type="primary", use_container_width=True):
                        with st.spinner("Deleting..."):
                            result = api.delete_chat_session(session_id, token)
                        if "error" not in result:
                            st.success("🗑️ Chat deleted!")
                            st.session_state.refresh_sessions = True
                            st.session_state.open_menus.discard(session_id)
                            st.session_state.confirm_deletes.discard(session_id)
                            st.session_state.pending_renames.pop(session_id, None)
                            if st.session_state.chat_session_id == session_id:
                                st.session_state.chat_session_id = None
                                st.session_state.chat_history = []
                            st.rerun()
                        else:
                            st.error(f"❌ Error: {result.get('error')}")
                
                with d_col2:
                    if st.button("❌ No", key=f"delete_cancel_{session_id}", use_container_width=True):
                        st.session_state.confirm_deletes.discard(session_id)
                        st.rerun()

def render_chat_messages(api, token):
    for msg_index, msg in enumerate(st.session_state.chat_history):
        with st.chat_message(msg["role"], avatar="👤" if msg["role"] == "user" else "🤖"):
            st.markdown(msg["content"])
            retrieved_docs = msg.get("retrieved")
            if retrieved_docs and isinstance(retrieved_docs, list) and len(retrieved_docs) > 0:
                render_sources_component(retrieved_docs)
            if msg["role"] == "assistant" and len(msg["content"]) > 50:
                render_message_actions(api, token, msg["content"], msg_index)

def handle_user_input(api, token, prompt, selected_file_id, sessions):
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Analyzing documents..."):
            response = api.query_rag(
                query=prompt,
                token=token,
                file_id=selected_file_id,
                session_id=st.session_state.chat_session_id
            )

        if "error" in response:
            st.error(f"Error: {response['error']}")
            return

        answer = response.get("answer", "")
        retrieved = response.get("retrieved", [])
        st.session_state.chat_session_id = response.get("session_id")
        
        st.markdown(answer)
        if retrieved and isinstance(retrieved, list) and len(retrieved) > 0:
            render_sources_component(retrieved)

        new_assistant_message = {
            "role": "assistant", 
            "content": answer,
            "retrieved": retrieved
        }
        st.session_state.chat_history.append(new_assistant_message)
        render_message_actions(api, token, answer, len(st.session_state.chat_history) - 1)

        # SENIOR ENG FIX: Force session refresh if it's a new conversation
        if not sessions or response.get("session_id") not in [s['id'] for s in sessions]:
            st.session_state.refresh_sessions = True # <--- THIS LINE FIXES YOUR ISSUE
            st.rerun()

def render_message_actions(api, token, message_content, message_index):
    clean_text = re.sub(r"^\*\*.*?:\*\*\s*\n\n", "", message_content).strip()
    if len(clean_text) > 50:
        with st.container():
            col1, col2, col3 = st.columns([1, 1, 4])
            if col1.button("✨ Summarize", key=f"summarize_{message_index}"):
                with st.spinner("Summarizing..."):
                    res = api.summarize_text(clean_text, token)
                if "error" in res:
                    st.error(res["error"])
                else:
                    summary = res.get("summary", "")
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": f"**📝 Summary of previous response:**\n\n{summary}"
                    })
                    st.rerun()

            with col2:
                with st.popover("🎨 Format"):
                    fmt = st.radio("Style", ["markdown", "bullet points", "table", "json"], key=f"fmt_radio_{message_index}")
                    if st.button("Apply Format", key=f"fmt_btn_{message_index}"):
                        with st.spinner("Formatting..."):
                            res = api.format_response(clean_text, token, fmt)
                        if "error" in res:
                            st.error(res["error"])
                        else:
                            st.session_state.chat_history.append({
                                "role": "assistant",
                                "content": f"**Formatted previous response as {fmt}:**\n\n{res.get('formatted_text')}"
                            })
                            st.rerun()

def render_sources_component(docs):
    if not docs or not isinstance(docs, list) or len(docs) == 0:
        return
    try:
        with st.expander(f"📚 Referenced Sources ({len(docs)})", expanded=False):
            num_cols = min(3, len(docs))
            cols = st.columns(num_cols)
            for i, doc in enumerate(docs):
                col = cols[i % num_cols]
                if isinstance(doc, dict):
                    meta = doc.get("metadata") or doc.get("meta") or {}
                    filename = meta.get("filename", "Unknown File")
                    score = doc.get("score", 0)
                    text_content = doc.get("text", "") or doc.get("content", "")
                else:
                    filename = "Unknown File"
                    score = 0
                    text_content = str(doc)
                text_preview = text_content[:100].replace("\n", " ") + "..."
                with col:
                    st.markdown(f"""
                    <div class="source-card">
                        <div style="font-weight: 600; font-size: 0.9rem; color: #333;">Source {i+1}</div>
                        <div style="font-size: 0.8rem; color: #666; margin-bottom: 8px;" title="{filename}">📄 {filename}</div>
                        <div style="font-size: 0.85rem; color: #444; overflow: hidden; height: 60px;">"{text_preview}"</div>
                        <div style="font-size: 0.75rem; color: #2e7d32; font-weight: bold; margin-top: 8px;">Relevance: {score:.2f}</div>
                    </div>
                    """, unsafe_allow_html=True)
    except Exception:
        pass