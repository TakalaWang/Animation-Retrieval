import os
import time
import logging
from dotenv import load_dotenv
import google.genai as genai
from concurrent.futures import ThreadPoolExecutor, as_completed

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# 讀多個 API Key
GEMINI_API_KEYS = [k.strip() for k in os.getenv("GEMINI_API_KEY", "").split(",") if k.strip()]
if not GEMINI_API_KEYS:
    raise RuntimeError("請先設定 GEMINI_API_KEY")

# 控制同一個 key 底下同時刪幾個檔
MAX_WORKERS_PER_KEY = 8  # 你可以調大或調小


def delete_one_file(client: genai.Client, file_name: str):
    """刪一個檔案，失敗不丟出到外面"""
    try:
        client.files.delete(name=file_name)
        logging.info(f"🗑️ 刪除 {file_name}")
    except Exception as e:
        logging.warning(f"⚠️ 刪除失敗 {file_name}: {e}")


def delete_all_files_for_key(api_key: str):
    prefix = f"[{api_key[:10]}...]"
    client = genai.Client(api_key=api_key)

    # 1) 列出檔案
    try:
        files = client.files.list()
    except Exception as e:
        logging.error(f"{prefix} 無法列出檔案：{e}")
        return

    if not files:
        logging.info(f"{prefix} ✅ 沒有可刪除的檔案")
        return

    logging.info(f"{prefix} 找到 {len(files)} 個檔案，準備刪除（並行）...")

    # 2) 並行刪除
    with ThreadPoolExecutor(max_workers=MAX_WORKERS_PER_KEY) as executor:
        futures = []
        for f in files:
            futures.append(executor.submit(delete_one_file, client, f.name))

        # 等全部做完（順便吃掉 exception）
        for _ in as_completed(futures):
            pass

    logging.info(f"{prefix} ✅ 這個 key 底下的檔案都處理完了")


def main():
    # 如果你想「多個 key 也一起並行」可以再包一層 ThreadPool
    # 這裡先簡單：逐個 key 處理，已經有檔案層級的並行了
    for key in GEMINI_API_KEYS:
        delete_all_files_for_key(key)
        # 視情況稍微休息，避免真的打太兇
        time.sleep(0.5)


if __name__ == "__main__":
    main()
