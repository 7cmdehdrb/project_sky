import os
import requests
import tqdm

install_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "resource")

# 설정값
server_url = "http://7cmdehdrb.iptime.org:13670/share/1S7XDEgO"
server_url = "http://7cmdehdrb.iptime.org:13670/api/public/dl/1S7XDEgO"
filename = "best_model.pth"

# 파일 다운로드
local_filepath = os.path.join(install_path, filename)

# 파일이 이미 존재하는지 확인
print(os.path.join(install_path, filename))
if os.path.exists(os.path.join(install_path, filename)):
    print(f"{filename} already exists in {install_path}. Stop downloading.")

else:
    print(f"Downloading {filename} from {server_url} ...")
    response: requests.models.Response = requests.get(server_url, stream=True)

    if response.status_code == 200:
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

        # 압축 파일이면 풀기 (옵션)
        if filename.endswith(".zip"):
            import zipfile

            with zipfile.ZipFile(local_filepath, "r") as zip_ref:
                zip_ref.extractall(install_path)
            print(f"Extracted {filename} to {install_path}")

        # 실행 가능한 파일이면 권한 부여
        if filename.endswith((".sh", ".bin")):
            os.chmod(local_filepath, 0o755)
            print(f"Set executable permissions for {local_filepath}")

    else:
        print(f"Failed to download {filename}, Status Code: {response.status_code}")


print("Installation complete.")
