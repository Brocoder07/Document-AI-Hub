import os
import urllib.request
import zipfile
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_3gpp_specs():
    out_dir = os.path.join("data", "documents")
    os.makedirs(out_dir, exist_ok=True)
    
    specs = {
        "TS_23.501": "https://www.3gpp.org/ftp/Specs/archive/23_series/23.501/23501-h60.zip",
        "TS_38.300": "https://www.3gpp.org/ftp/Specs/archive/38_series/38.300/38300-h20.zip",
        "TS_23.502": "https://www.3gpp.org/ftp/Specs/archive/23_series/23.502/23502-h60.zip",
        "TS_29.500": "https://www.3gpp.org/ftp/Specs/archive/29_series/29.500/29500-h60.zip"
    }

    headers = {'User-Agent': 'Mozilla/5.0'}

    for name, url in specs.items():
        logger.info(f"Downloading {name} from {url}...")
        zip_path = os.path.join(out_dir, f"{name}.zip")
        
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req) as response, open(zip_path, 'wb') as out_file:
                data = response.read()
                out_file.write(data)
                
            logger.info(f"Extracting {name}...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # 3GPP usually puts a single .docx inside
                for file_info in zip_ref.infolist():
                    if file_info.filename.endswith('.docx') or file_info.filename.endswith('.doc'):
                        # rename it to our name
                        extracted_name = f"{name}.docx"
                        extracted_path = os.path.join(out_dir, extracted_name)
                        
                        with open(extracted_path, "wb") as f:
                            f.write(zip_ref.read(file_info.filename))
                        
                        logger.info(f"✅ Saved {extracted_name} to {out_dir}")
            
            # Clean up the zip file
            os.remove(zip_path)
            
        except Exception as e:
            logger.error(f"❌ Failed to download {name}: {e}")

if __name__ == "__main__":
    download_3gpp_specs()
