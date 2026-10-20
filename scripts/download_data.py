import os
import urllib.request
import zipfile

URL = "https://s3-us-west-2.amazonaws.com/ailab-data-public/cmapss/NASA-CMAPSS.zip"
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def download():
    os.makedirs(DATA_DIR, exist_ok=True)
    zip_path = os.path.join(DATA_DIR, "NASA-CMAPSS.zip")

    if not os.path.exists(zip_path):
        print(f"Downloading C-MAPSS dataset ({URL})...")
        urllib.request.urlretrieve(URL, zip_path)
        print("Download complete.")
    else:
        print("ZIP already exists, skipping download.")

    print("Extracting...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(DATA_DIR)
    print(f"Extracted to {DATA_DIR}")

    expected = [
        "train_FD001.txt", "test_FD001.txt", "RUL_FD001.txt",
        "train_FD002.txt", "test_FD002.txt", "RUL_FD002.txt",
        "train_FD003.txt", "test_FD003.txt", "RUL_FD003.txt",
        "train_FD004.txt", "test_FD004.txt", "RUL_FD004.txt"
    ]

    for fname in expected:
        fpath = os.path.join(DATA_DIR, fname)
        if os.path.exists(fpath):
            print(f"  {fname} OK")
        else:
            for root, dirs, files in os.walk(DATA_DIR):
                if fname in files:
                    src = os.path.join(root, fname)
                    dst = os.path.join(DATA_DIR, fname)
                    os.rename(src, dst)
                    print(f"  {fname} moved into place")
                    break
            else:
                print(f"  {fname} NOT FOUND")

    print("\nDone. Expected structure:")
    print("  data/train_FD001.txt, test_FD001.txt, RUL_FD001.txt")
    print("  data/train_FD002.txt, test_FD002.txt, RUL_FD002.txt")
    print("  data/train_FD003.txt, test_FD003.txt, RUL_FD003.txt")
    print("  data/train_FD004.txt, test_FD004.txt, RUL_FD004.txt")


if __name__ == "__main__":
    download()
