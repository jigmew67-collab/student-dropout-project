"""Download the original OULAD release from its Figshare record."""
from hashlib import md5
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "data" / "raw" / "anonymisedData.zip"
URL = "https://ndownloader.figshare.com/files/8606371"
EXPECTED_MD5 = "7412686fd77cf0e0ee1e8c3e9b354308"

def main():
    DEST.parent.mkdir(parents=True, exist_ok=True)
    digest = md5()
    with urlopen(URL, timeout=90) as response, DEST.open("wb") as output:
        while chunk := response.read(1024 * 1024):
            output.write(chunk)
            digest.update(chunk)
    if digest.hexdigest() != EXPECTED_MD5:
        DEST.unlink(missing_ok=True)
        raise RuntimeError("OULAD archive checksum mismatch; partial or changed data removed")
    print(f"Downloaded verified OULAD archive ({DEST.stat().st_size:,} bytes): {DEST}")

if __name__ == "__main__":
    main()
