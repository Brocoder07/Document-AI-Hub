import requests
import os

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")

class APIClient:
    def __init__(self):
        self.base_url = API_BASE_URL

    def _headers(self, token):
        return {"Authorization": f"Bearer {token}"}

    def _handle_error(self, e):
        try:
            return {"error": e.response.json().get('detail', str(e))}
        except:
            return {"error": str(e)}

    # --- AUTH ---
    def login(self, email, password):
        try:
            res = requests.post(
                f"{self.base_url}/auth/token", 
                data={"username": email, "password": password},
                timeout=30
            )
            res.raise_for_status()
            return res.json()
        except Exception as e: return self._handle_error(e)

    def register(self, username, email, full_name, password, role):
        try:
            res = requests.post(f"{self.base_url}/auth/register", json={
                "username": username, "email": email, "full_name": full_name, 
                "password": password, "role": role
            })
            res.raise_for_status()
            return res.json()
        except Exception as e: return self._handle_error(e)

    # --- USERS ---
    def get_current_user(self, token):
        try:
            res = requests.get(f"{self.base_url}/users/me", headers=self._headers(token))
            res.raise_for_status()
            return res.json()
        except Exception as e: return self._handle_error(e)

    def update_user(self, token, data):
        try:
            res = requests.patch(f"{self.base_url}/users/me", json=data, headers=self._headers(token))
            res.raise_for_status()
            return res.json()
        except Exception as e: return self._handle_error(e)

    def delete_user(self, token):
        try:
            requests.delete(f"{self.base_url}/users/me", headers=self._headers(token))
            return True
        except: return False

    # --- FILES ---
    def get_user_files(self, token):
        try:
            res = requests.get(f"{self.base_url}/upload/files", headers=self._headers(token))
            res.raise_for_status()
            return res.json().get("files", [])
        except: return []

    def upload_file(self, file_obj, token):
        try:
            res = requests.post(f"{self.base_url}/upload/file", headers=self._headers(token), files={"file": file_obj})
            res.raise_for_status()
            return res.json()
        except Exception as e: return self._handle_error(e)

    def delete_file(self, file_id, token):
        try:
            requests.delete(f"{self.base_url}/upload/file/{file_id}", headers=self._headers(token))
            return True
        except: return False

    # --- PROCESSING TOOLS ---
    def ocr_extract(self, file_id, token):
        try:
            res = requests.post(f"{self.base_url}/ocr/extract", json={"file_id": file_id}, headers=self._headers(token), timeout=60)
            res.raise_for_status()
            return res.json()
        except Exception as e: return self._handle_error(e)

    def transcribe_audio(self, file_id, token):
        try:
            res = requests.post(f"{self.base_url}/transcription/audio", json={"file_id": file_id}, headers=self._headers(token), timeout=120)
            res.raise_for_status()
            return res.json()
        except Exception as e: return self._handle_error(e)

    def summarize_text(self, text, token, method="extractive"):
        """
        Summarize text using form data (not JSON) to handle newlines properly.
        """
        if not text or not text.strip():
            return {"error": "No text provided to summarize"}
        
        try:
            # Prepare form data
            form_data = {
                "text": str(text).strip(),
                "method": str(method).lower()
            }
            
            # Remove empty values
            form_data = {k: v for k, v in form_data.items() if v}
            
            res = requests.post(
                f"{self.base_url}/summarize/text", 
                data=form_data,  # Form data, not JSON
                headers=self._headers(token),
                timeout=60
            )
            res.raise_for_status()
            return res.json()
            
        except requests.exceptions.Timeout:
            return {"error": "Summarization request timed out"}
        except requests.exceptions.ConnectionError:
            return {"error": "Cannot connect to server"}
        except Exception as e: 
            return self._handle_error(e)

    # --- UTILITIES ---
    def generate_embeddings(self, text, token):
        try:
            res = requests.post(f"{self.base_url}/embeddings/generate", json={"text": text}, headers=self._headers(token))
            res.raise_for_status()
            return res.json()
        except Exception as e: return self._handle_error(e)

    def similarity_search(self, query, token, top_k=5):
        try:
            res = requests.post(f"{self.base_url}/search/similarity", json={"query": query, "top_k": top_k}, headers=self._headers(token))
            res.raise_for_status()
            return res.json()
        except Exception as e: return self._handle_error(e)

    def format_response(self, text, token, fmt="markdown"):
        try:
            res = requests.post(f"{self.base_url}/format/response", json={"text": text, "format": fmt}, headers=self._headers(token), timeout=30)
            res.raise_for_status()
            return res.json()
        except Exception as e: return self._handle_error(e)

    # --- RAG & HISTORY ---
    def query_rag(self, query, token, file_id=None, session_id=None):
        """
        UPDATED: Removed 'mode' parameter entirely.
        The backend now determines the mode based on the user's role.
        """
        payload = {"query": query, "session_id": session_id}
        if file_id: payload["file_id"] = file_id
        try:
            res = requests.post(f"{self.base_url}/rag/answer", json=payload, headers=self._headers(token), timeout=60)
            res.raise_for_status()
            return res.json()
        except Exception as e: return self._handle_error(e)

    def get_chat_sessions(self, token):
        try:
            res = requests.get(f"{self.base_url}/rag/history", headers=self._headers(token))
            res.raise_for_status()
            return res.json()
        except: return []

    def get_session_messages(self, session_id, token):
        try:
            res = requests.get(f"{self.base_url}/rag/history/{session_id}", headers=self._headers(token))
            res.raise_for_status()
            return res.json()
        except: return []

    def update_chat_title(self, session_id: str, new_title: str, token: str):
        try:
            res = requests.patch(
                f"{self.base_url}/rag/history/{session_id}/title",
                json={"title": new_title},
                headers=self._headers(token)
            )
            res.raise_for_status()
            return res.json()
        except Exception as e:
            return self._handle_error(e)

    def delete_chat_session(self, session_id: str, token: str):
        try:
            res = requests.delete(
                f"{self.base_url}/rag/history/{session_id}",
                headers=self._headers(token)
            )
            res.raise_for_status()
            return {"success": True, "message": "Chat session deleted"}
        except Exception as e:
            return self._handle_error(e)