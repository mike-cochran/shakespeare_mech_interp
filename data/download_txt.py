import os
import zipfile

import requests

# Data location(s)
RAW_PATH = "./folger"
os.makedirs(RAW_PATH, exist_ok=True)
RAW_URL = "https://flgr.sh/txtfssAlltxt?_ga=2.30930617.761684405.1773926679-1180367625.1773375445"
ZIP_FILE = os.path.join(RAW_PATH, "shakespeares-works.zip")
EXTRACT_FOLDER = os.path.join(RAW_PATH, "shakespeares-works_TXT_FolgerShakespeare")

# Check if data already downloaded otherwise download txt from Folger website
if os.path.exists(ZIP_FILE):
    print("Shakespeare zip file already downloaded")
else:
    r = requests.get(RAW_URL)
    with open(ZIP_FILE, "wb") as f:
        f.write(r.content)
    print(f"Downloaded {ZIP_FILE}")

# Check if data already unzipped
if os.path.isdir(EXTRACT_FOLDER) and len(os.listdir(EXTRACT_FOLDER)) > 0:
    print("Shakespeare data already unzipped")
else:
    print(f"Extracting files into {EXTRACT_FOLDER}...")
    os.makedirs(EXTRACT_FOLDER, exist_ok=True)
    with zipfile.ZipFile(ZIP_FILE, "r") as zip_ref:
        zip_ref.extractall(EXTRACT_FOLDER)
    print(f"Shakespeare extraction complete {ZIP_FILE}")
