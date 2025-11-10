# test_single.py
import os
import json
import time
import shutil
import tempfile
from pathlib import Path
from dotenv import load_dotenv
import google.genai as genai

load_dotenv()
GEMINI_KEY = os.getenv("GEMINI_API_KEY").split(",")[0]

def upload_with_unicode_fix(client, path: str):
    """處理中文檔名的上傳"""
    p = Path(path)
    
    # 檢查檔名是否包含非 ASCII 字元
    try:
        p.name.encode("ascii")
        upload_path = str(p)
    except UnicodeEncodeError:
        # 複製到臨時檔案,使用純 ASCII 檔名
        print(f"⚠️ 檔名包含中文,複製到臨時檔案")
        tmp = Path(tempfile.gettempdir()) / f"tmp_{int(time.time()*1000)}{p.suffix}"
        shutil.copy2(p, tmp)
        upload_path = str(tmp)
        print(f"臨時檔案: {upload_path}")
    
    return client.files.upload(file=upload_path)

def test_upload_and_generate():
    client = genai.Client(api_key=GEMINI_KEY)
    
    # 找你實際的 segment 檔案
    test_file = "cache_gemini_video/videos/HIGH_CARD_至高之牌_2/segment_HIGH_CARD_至高之牌_2_14_seg0.mp4"
    
    # 或是用通配符找第一個可用的
    from glob import glob
    segments = glob("cache_gemini_video/videos/*/segment_*.mp4")
    if segments:
        test_file = segments[0]
        print(f"使用檔案: {test_file}")
    
    if not Path(test_file).exists():
        print("❌ 找不到測試檔案")
        print("可用的檔案:")
        for f in segments[:5]:
            print(f"  - {f}")
        return
    
    file_size_mb = Path(test_file).stat().st_size / 1024 / 1024
    print(f"📁 檔案大小: {file_size_mb:.2f}MB")
    
    # 測試上傳
    print("\n📤 開始上傳...")
    try:
        file_obj = upload_with_unicode_fix(client, test_file)
        print(f"✅ 上傳成功: {file_obj.name}")
        print(f"📊 狀態: {file_obj.state.name}")
        
        # 等待處理
        wait_count = 0
        while file_obj.state.name == "PROCESSING":
            wait_count += 1
            print(f"⏳ 等待處理中... ({wait_count * 5}秒)")
            time.sleep(5)
            file_obj = client.files.get(name=file_obj.name)
            
            if wait_count > 60:  # 超過 5 分鐘
                print("❌ 處理超時")
                return
        
        print(f"📊 最終狀態: {file_obj.state.name}")
        
        if file_obj.state.name == "ACTIVE":
            print(f"✅ URI: {file_obj.uri}")
            
            # 測試生成
            print("\n🤖 開始生成 queries...")
            try:
                from segment_processor import generate_segment_queries
                
                start_time = time.time()
                queries = generate_segment_queries(client=client, file_uri=file_obj.uri)
                elapsed = time.time() - start_time
                
                print(f"✅ 生成成功! (耗時: {elapsed:.1f}秒)")
                print("\n生成的 queries:")
                print(json.dumps(queries, indent=2, ensure_ascii=False))
                
            except Exception as e:
                print(f"❌ 生成失敗: {e}")
                print(f"錯誤類型: {type(e).__name__}")
                import traceback
                traceback.print_exc()
        else:
            print(f"❌ 處理失敗: {file_obj.state.name}")
            
    except Exception as e:
        print(f"❌ 上傳錯誤: {e}")
        print(f"錯誤類型: {type(e).__name__}")
        
        # 檢查是否是 503 錯誤
        error_str = str(e)
        if "503" in error_str:
            print("\n⚠️ 這是 503 Service Unavailable 錯誤")
            print("可能原因:")
            print("1. Gemini API 後端過載")
            print("2. 你的 API key 達到 rate limit")
            print("3. 該模型暫時不可用")
        elif "429" in error_str:
            print("\n⚠️ 這是 429 Too Many Requests 錯誤")
            print("你的 API key 達到請求限制,需要等待")
        elif "403" in error_str or "PERMISSION_DENIED" in error_str:
            print("\n⚠️ 這是 403 Permission Denied 錯誤")
            print("你的 API key 可能被停用或沒有權限")
        
        import traceback
        traceback.print_exc()

def test_simple_text():
    """測試簡單的文字生成,確認 API key 可用"""
    client = genai.Client(api_key=GEMINI_KEY)
    print("🧪 測試簡單的文字生成...")
    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash-exp",
            contents="Say 'Hello, I am working!' in one sentence."
        )
        print(f"✅ 文字生成成功: {response.text}")
        return True
    except Exception as e:
        print(f"❌ 文字生成失敗: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("Gemini API 測試")
    print("=" * 60)
    
    # 先測試文字生成
    if test_simple_text():
        print("\n" + "=" * 60)
        print("開始測試視頻處理")
        print("=" * 60)
        test_upload_and_generate()
    else:
        print("\n❌ 連文字生成都失敗,請檢查 API key")