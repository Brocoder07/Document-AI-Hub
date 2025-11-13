from pathlib import Path
from app.core.config import settings
import uuid

def save_upload(file_obj, filename: str) -> str:
    uid = uuid.uuid4().hex
    out_name = f"{uid}_{filename}"
    dest = Path(settings.UPLOAD_DIR) / out_name
    with open(dest, "wb") as f:
        f.write(file_obj.read())
    return str(dest)
