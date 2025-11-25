import streamlit as st
from api_client import APIClient
import time

def render_documents_page(api: APIClient):
    st.title("📂 Document Library")
    token = st.session_state.access_token

    # 1. Upload Section
    with st.expander("⬆️ Upload New File", expanded=True):
        f = st.file_uploader(
            "Choose file", 
            type=[
                "pdf", "png", "jpg", "jpeg",  # Images/Docs
                "mp3", "wav", "mp4", "m4a",   # Audio/Video
                "txt", "md", "doc", "docx"    # Text/Office
            ]
        )
        if f and st.button("Upload"):
            with st.spinner("Uploading & Indexing..."):
                res = api.upload_file(f, token)
            if "error" in res: st.error(res["error"])
            else: 
                st.success("Uploaded successfully!")
                st.rerun()

    st.divider()

    # 2. File List & Actions
    files = api.get_user_files(token)
    if not files:
        st.info("No files found.")
        return

    for file in files:
        with st.container():
            c1, c2, c3, c4 = st.columns([3, 1, 2, 2])
            c1.markdown(f"**{file['filename']}**")
            c2.caption(file['file_type'].upper())
            c3.caption(file['upload_date'])
            
            # Action Buttons
            with c4:
                col_a, col_b = st.columns(2)
                
                # Delete
                if col_a.button("🗑️", key=f"del_{file['file_id']}"):
                    if api.delete_file(file['file_id'], token):
                        st.toast("File deleted")
                        st.rerun()
                    else:
                        st.error("Failed to delete")
                
                ftype = file['file_type'].lower()
                
                # Logic: Redirect to Chat for results
                if ftype in ['mp3', 'wav', 'mp4', 'm4a']:
                    if col_b.button("🎙️ Transcribe", key=f"trans_{file['file_id']}"):
                        handle_processing(api, token, file['file_id'], "transcription")
                
                elif ftype in ['png', 'jpg', 'jpeg', 'pdf']:
                    if col_b.button("🔍 OCR", key=f"ocr_{file['file_id']}"):
                        handle_processing(api, token, file['file_id'], "ocr")

                elif ftype in ['txt', 'md', 'doc', 'docx']:
                    if col_b.button("📝 Extract", key=f"ext_{file['file_id']}"):
                        handle_processing(api, token, file['file_id'], "extraction")

            st.markdown("---")

def handle_processing(api, token, file_id, mode):
    """
    Runs the processing task and redirects the result to the Chat Interface.
    """
    with st.spinner(f"Running {mode}... Please wait."):
        if mode == "transcription":
            res = api.transcribe_audio(file_id, token)
            content_key = "transcription"
        else:
            res = api.ocr_extract(file_id, token)
            content_key = "extracted_text"
    
    if "error" in res:
        st.error("Operation Failed")
        with st.expander("See Error Details"):
            st.write(res["error"])
    else:
        text_result = res.get(content_key, "")
        
        # 1. Switch to Chat Tab
        st.session_state.nav_section = "Chat"
        
        # 2. Inject Result into Chat History
        success_msg = f"✅ **{mode.upper()} Complete!** Here is the content:\n\n{text_result}"
        
        # Create session if needed (Frontend only for instant UI)
        if not st.session_state.chat_history:
            st.session_state.chat_history = []
            
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": success_msg
        })
        
        st.toast(f"{mode.title()} successful! Redirecting to chat...")
        time.sleep(0.5)
        st.rerun()