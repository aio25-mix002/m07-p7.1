import os
import shutil
import gdown
import zipfile
from pathlib import Path

FILE_ID = '141MgG4CC7XffVH32hQy7lQ0PKCafskji'
OUTPUT_ZIP = 'HMDB51.zip'
DATA_DIR = 'hmdb51_data'

def download_and_extract():
   
    if os.path.exists(DATA_DIR) and os.path.isdir(DATA_DIR):
        if len(os.listdir(DATA_DIR)) > 0:
            print(f"Dataset directory '{DATA_DIR}' already exists and is not empty. Skipping download.")
            return

    print(f"Downloading {OUTPUT_ZIP} from Google Drive...")
    url = f'https://drive.google.com/uc?id={FILE_ID}'
    
    gdown.download(url, OUTPUT_ZIP, quiet=False)

    # Giải nén
    if not os.path.exists(OUTPUT_ZIP):
        print("Download failed!")
        return

    print(f"Extracting {OUTPUT_ZIP}...")
    
    # Tạo thư mục đích
    Path(DATA_DIR).mkdir(parents=True, exist_ok=True)
    
    try:
        with zipfile.ZipFile(OUTPUT_ZIP, 'r') as zip_ref:
           
            zip_ref.extractall(DATA_DIR)
        print("Extraction complete.")
        
        # Dọn dẹp: Xóa file zip để tiết kiệm ổ cứng
        os.remove(OUTPUT_ZIP)
        print(f"Removed temporary file {OUTPUT_ZIP}")
        

        sub_dirs = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
        if len(sub_dirs) == 1 and sub_dirs[0].lower() in ['hmdb51', 'data']:
            nested_dir = os.path.join(DATA_DIR, sub_dirs[0])
            print(f"Moving files from nested directory: {nested_dir}")
            for item in os.listdir(nested_dir):
                shutil.move(os.path.join(nested_dir, item), DATA_DIR)
            os.rmdir(nested_dir)

        print(f"Data is ready at: {os.path.abspath(DATA_DIR)}")

    except zipfile.BadZipFile:
        print("Error: The downloaded file is not a valid zip file.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    download_and_extract()