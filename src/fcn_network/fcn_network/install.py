import os
import requests
import tqdm

install_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "resource")


def download_file(
    server_url: str,
    filename: str,
    install_path: str,
    extract_subdir: str = None,
) -> None:
    local_filepath = os.path.join(install_path, filename)

    print(local_filepath)
    if os.path.exists(local_filepath):
        print(f"{filename} already exists in {install_path}. Stop downloading.")
        return

    print(f"Downloading {filename} from {server_url} ...")
    response: requests.models.Response = requests.get(server_url, stream=True)

    if response.status_code != 200:
        print(f"Failed to download {filename}, Status Code: {response.status_code}")
        return

    with open(local_filepath, "wb") as f:
        total_size = int(response.headers.get("content-length", 0))
        with tqdm.tqdm(
            total=total_size, unit="B", unit_scale=True, desc=filename
        ) as pbar:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

    print(f"Downloaded {filename} to {local_filepath}")

    if filename.endswith(".zip"):
        import zipfile

        extract_path = install_path
        if extract_subdir:
            extract_path = os.path.join(install_path, extract_subdir)
            os.makedirs(extract_path, exist_ok=True)

        with zipfile.ZipFile(local_filepath, "r") as zip_ref:
            zip_ref.extractall(extract_path)
        print(f"Extracted {filename} to {extract_path}")

    if filename.endswith((".sh", ".bin")):
        os.chmod(local_filepath, 0o755)
        print(f"Set executable permissions for {local_filepath}")


download_targets = [
    ("http://7cmdehdrb.iptime.org/api/public/dl/vo0W2ESb", "best_model.pth", None),
    ("http://7cmdehdrb.iptime.org/api/public/dl/W8AY4jlg", "best_model_45.pth", None),
    (
        "http://7cmdehdrb.iptime.org/api/public/dl/3j_RIFjA/",
        "exported_45.zip",
        "exported_45",
    ),
    (
        "http://7cmdehdrb.iptime.org/api/public/dl/_MUI8oCt/",
        "exported.zip",
        "exported",
    ),
]


for server_url, filename, extract_subdir in download_targets:
    download_file(server_url, filename, install_path, extract_subdir)

print("Installation complete.")
