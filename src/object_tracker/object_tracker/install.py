import os
import sys
import requests
import tqdm


def download_file(server_url: str, filename: str, install_path: str = None) -> None:
    """
    주어진 URL에서 파일을 다운로드합니다.

    Args:
        server_url (str): 파일이 위치한 서버 URL (예: "http://example.com/file.pt")
        filename (str): 저장할 파일 이름 (예: "model.pt")
        install_path (str, optional): 설치 경로. 지정하지 않으면 스크립트가 있는 디렉토리의 `resource` 폴더 사용.
    """
    if install_path is None:
        install_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "resource"
        )

    local_filepath = os.path.join(install_path, filename)

    # 디렉토리 없으면 생성
    os.makedirs(install_path, exist_ok=True)

    # 파일 존재 여부 확인
    if os.path.exists(local_filepath):
        print(f"{filename} already exists in {install_path}. Skipping download.")
        return

    print(f"Downloading {filename} from {server_url} ...")

    try:
        response: requests.models.Response = requests.get(server_url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))

        with open(local_filepath, "wb") as f:
            with tqdm.tqdm(
                total=total_size,
                unit="B",
                unit_scale=True,
                desc=filename,
                disable=False,
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
        if filename.endswith((".sh", ".bin", ".exe")):
            os.chmod(local_filepath, 0o755)
            print(f"Set executable permissions for {local_filepath}")

    except requests.RequestException as e:
        print(f"Failed to download {filename}: {e}", file=sys.stderr)


def main():
    # 기본 모델 다운로드 (기존 기능 유지)
    install_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "resource")

    download_file(
        server_url="http://7cmdehdrb.iptime.org/api/public/dl/N5F-j0YO",
        filename="best_hg.pt",
        install_path=install_path,
    )
    download_file(
        server_url="http://7cmdehdrb.iptime.org/api/public/dl/CKHsoSZl",
        filename="best_yolo45.pt",
        install_path=install_path,
    )

    # 필요 시 추가 파일도 여기에 등록
    # download_file("http://example.com/model2.onnx", "model2.onnx", install_path)


if __name__ == "__main__":
    main()
