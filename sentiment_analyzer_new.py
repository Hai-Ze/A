#!/usr/bin/env python3
# coding: utf-8
import os
import sys
import json
import time
import requests
import pandas as pd
import asyncio
import aiohttp
from datetime import datetime
import logging
from threading import Lock
import hashlib
import random

# Thiết lập logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("sentiment_analysis.log", encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Cấu hình đường dẫn
INPUT_FILE = r"C:\Users\taoda\Documents\Zalo Received Files\cawldata\Apple\apple_news_content.csv"  # Đường dẫn file CSV đầu vào
OUTPUT_FILE = r"C:\Users\taoda\Documents\Zalo Received Files\cawldata\Apple\apple_news_content1.csv"  # Đường dẫn file đầu ra
CHECKPOINT_DIR = "checkpoints"  # Thư mục lưu các checkpoint

# Cấu hình API
MAX_TOKENS = 800  # Tăng token tối đa để đảm bảo đủ cho phân tích chi tiết
MODEL = "mistral-small-latest"  # Sử dụng model nhỏ hơn, nhanh hơn
TEMPERATURE = 0.0  # Giảm xuống 0 để tăng độ ổn định
REQUEST_TIMEOUT = 60  # Thời gian timeout cho mỗi request (giây)
CHECKPOINT_INTERVAL = 20  # Lưu checkpoint sau mỗi bao nhiêu tin tức
MAX_CONTENT_LENGTH = 4000  # Độ dài tối đa của nội dung để tiết kiệm token
MAX_RETRIES = 5  # Số lần thử lại tối đa cho mỗi request
RATE_LIMIT_DELAY = 1.05  # Thời gian giữa các request từ cùng một API key (giây)

# Cổ phiếu mục tiêu để phân tích - thay đổi khi cần
TARGET_STOCK = "NVIDIA"  # Cổ phiếu mục tiêu
STOCK_SYMBOL = "NVDA"    # Mã cổ phiếu

# Danh sách các từ khóa liên quan đến cổ phiếu mục tiêu
TARGET_KEYWORDS = [
    "NVIDIA", "NVDA", "Jensen Huang", "GPU", "RTX", "GeForce", "Data center", "AI chip",
    "RTX", "tensor core", "CUDA", "HGX", "DGX", "Drive", "Jetson", "Blackwell", "Hopper",
    "Lovelace", "semiconductor", "chip", "graphics", "computing", "deep learning", "gaming",
    "artificial intelligence", "Mellanox", "Arm", "data center", "cloud gaming", "cloud computing"
]

# Danh sách các cột kết quả từ phân tích cảm xúc
SENTIMENT_COLUMNS = [
    'sentiment_label', 'sentiment_score', 'impact_score', 'relevance_score', 'confidence', 
    'target_mentioned', 'direct_impact', 'short_term_score', 'medium_term_score', 
    'topic_financial_performance', 'topic_corporate_changes', 'topic_market_conditions', 
    'topic_regulatory_news', 'topic_investor_sentiment', 'topic_competition',
    'topic_technology', 'topic_ai_sector',
    'prominence', 'timeliness'
]

# Bộ nhớ đệm cho kết quả
result_cache = {}
cache_lock = Lock()

# Trạng thái API key
api_key_locks = {}
api_key_last_used = {}

# Đọc API keys từ file .env và biến môi trường
def get_api_keys():
    """Lấy danh sách tất cả API key khả dụng"""
    keys = []
    # Danh sách tên các API key cần tìm
    key_names = ["MISTRAL_API_KEY", "MISTRAL_API_KEY_ZES", "MISTRAL_API_KEY_NPQ", "MISTRAL_API_KEY_ZES2"]
    
    # Đọc từ file .env
    env_path = os.path.join(os.getcwd(), '.env')
    env_vars = {}
    
    if os.path.exists(env_path):
        with open(env_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip() and '=' in line and not line.startswith('#'):
                    key, value = line.strip().split('=', 1)
                    env_vars[key] = value.strip('"\'')
    
    # Thu thập các API key từ file .env và biến môi trường
    for key_name in key_names:
        # Thử từ .env trước
        if key_name in env_vars and env_vars[key_name]:
            keys.append(env_vars[key_name])
        # Nếu không có trong .env, thử từ biến môi trường
        elif os.environ.get(key_name):
            keys.append(os.environ.get(key_name))
    
    # Loại bỏ các key rỗng
    keys = [k for k in keys if k]
    
    # Khởi tạo các khóa và thời gian sử dụng cho mỗi API key
    global api_key_locks, api_key_last_used
    for key in keys:
        api_key_locks[key] = asyncio.Lock()
        api_key_last_used[key] = 0
    
    return keys

# Tiền xử lý nội dung để giảm kích thước
def preprocess_content(content, max_length=MAX_CONTENT_LENGTH):
    """Cắt và tối ưu nội dung để giảm kích thước"""
    if not content or not isinstance(content, str):
        return ""
    
    if len(content) <= max_length:
        return content
    
    # Cắt dữ liệu thông minh: Lấy phần đầu và phần cuối, bỏ phần giữa
    head_size = int(max_length * 0.7)  # 70% đầu tiên
    tail_size = max_length - head_size  # 30% cuối cùng
    
    head = content[:head_size]
    tail = content[-tail_size:] if len(content) > tail_size else ""
    
    return head + "..." + tail

# Tạo mã hash cho văn bản để làm khóa bộ nhớ đệm
def get_cache_key(text):
    """Tạo mã hash cho văn bản để sử dụng làm khóa cho bộ nhớ đệm"""
    return hashlib.md5(text.encode('utf-8')).hexdigest()

# Phân tích một tin tức sử dụng API key cố định với prompt tối ưu cho NVIDIA
async def analyze_news(session, api_key, header, content, date, news_idx):
    """Phân tích một tin tức bằng API key cố định"""
    # Tiền xử lý nội dung
    processed_content = preprocess_content(content)
    
    # Tạo text kết hợp
    combined_text = f"Tiêu đề: {header}\n\nNội dung: {processed_content}\n\nNgày đăng: {date}"
    
    # Kiểm tra bộ nhớ đệm
    cache_key = get_cache_key(combined_text)
    with cache_lock:
        if cache_key in result_cache:
            logger.info(f"Tin {news_idx}: Đã tìm thấy trong bộ nhớ đệm")
            return result_cache[cache_key]
    
    # Tạo prompt tối ưu cho cổ phiếu cụ thể (NVIDIA)
    prompt = f"""Phân tích tin tức tài chính sau đây để xác định mức độ liên quan và tác động đến cổ phiếu {TARGET_STOCK} (mã: {STOCK_SYMBOL}):

{combined_text}

Yêu cầu:
1. Xác định xem tin tức này có đề cập trực tiếp hoặc gián tiếp đến {TARGET_STOCK} không?
2. Đánh giá mức độ liên quan của tin tức đối với {TARGET_STOCK} (0.0-1.0).
3. Nếu có liên quan, phân tích cảm xúc và tác động tiềm năng đến giá cổ phiếu {STOCK_SYMBOL}.
4. Ngay cả khi tin tức về thị trường chung hoặc ngành công nghiệp, hãy đánh giá tác động gián tiếp đến {TARGET_STOCK}.
5. Nếu tin tức KHÔNG liên quan đến {TARGET_STOCK}, đặt "relevance_score" gần 0.0 và "impact_score" gần 0.0.

Trả về JSON với cấu trúc sau (KHÔNG giải thích thêm):
{{
  "sentiment_label": "POSITIVE/NEGATIVE/NEUTRAL/VERY_POSITIVE/VERY_NEGATIVE",
  "sentiment_score": -5.0 đến 5.0,
  "impact_score": -3.0 đến 3.0 (0 nếu không liên quan đến {TARGET_STOCK}),
  "relevance_score": 0.0 đến 1.0 (mức độ liên quan đến {TARGET_STOCK}),
  "target_mentioned": true/false (có đề cập trực tiếp đến {TARGET_STOCK} không),
  "direct_impact": true/false (có tác động trực tiếp đến {TARGET_STOCK} không),
  "confidence": 0.0 đến 1.0,
  "time_scores": {{
    "short_term_score": -3.0 đến 3.0 (tác động ngắn hạn đến {TARGET_STOCK}),
    "medium_term_score": -3.0 đến 3.0 (tác động trung hạn đến {TARGET_STOCK})
  }},
  "topic_weights": {{
    "financial_performance": 0.0 đến 1.0,
    "corporate_changes": 0.0 đến 1.0,
    "market_conditions": 0.0 đến 1.0,
    "regulatory_news": 0.0 đến 1.0,
    "investor_sentiment": 0.0 đến 1.0,
    "competition": 0.0 đến 1.0,
    "technology": 0.0 đến 1.0,
    "ai_sector": 0.0 đến 1.0
  }},
  "prominence": 1 đến 10,
  "timeliness": 1 đến 10
}}"""
    
    # Tạo request body
    request_data = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": f"Bạn là trợ lý AI chuyên về phân tích tin tức tài chính để đánh giá tác động đến cổ phiếu {TARGET_STOCK}. Nhiệm vụ của bạn là phân tích các tin tức, xác định mức độ liên quan và tác động đến {TARGET_STOCK}, đặc biệt chú ý phân biệt tin tức có liên quan thực sự với tin tức không liên quan."},
            {"role": "user", "content": prompt}
        ],
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS
    }
    
    # Headers
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    # Giới hạn tần suất API cho mỗi key
    async with api_key_locks[api_key]:
        # Kiểm tra và đợi nếu gọi API quá nhanh
        current_time = time.time()
        time_since_last_call = current_time - api_key_last_used.get(api_key, 0)
        if time_since_last_call < RATE_LIMIT_DELAY:
            wait_time = RATE_LIMIT_DELAY - time_since_last_call
            await asyncio.sleep(wait_time)
        
        # Cập nhật thời gian gọi cuối cùng
        api_key_last_used[api_key] = time.time()
        
        # Retry logic
        for retry in range(MAX_RETRIES):
            try:
                # Gửi request
                async with session.post(
                    "https://api.mistral.ai/v1/chat/completions",
                    headers=headers,
                    json=request_data,
                    timeout=REQUEST_TIMEOUT
                ) as response:
                    if response.status == 200:
                        response_json = await response.json()
                        
                        # Trích xuất nội dung
                        if 'choices' in response_json and len(response_json['choices']) > 0:
                            content = response_json['choices'][0]['message']['content']
                            
                            # Trích xuất JSON
                            result = extract_json(content)
                            if result:
                                # Lưu vào cache
                                with cache_lock:
                                    result_cache[cache_key] = result
                                return result
                    elif response.status == 429:  # Rate limit
                        # Nếu gặp rate limit, đợi và thử lại
                        logger.warning(f"Tin {news_idx}: Rate limit hit với API key {api_key[:5]}..., thử lại sau (lần {retry+1}/{MAX_RETRIES})")
                        await asyncio.sleep(3 * (retry + 1))  # Tăng thời gian đợi sau mỗi lần thử
                        continue
                    else:
                        error_text = await response.text()
                        logger.error(f"Tin {news_idx}: API error ({response.status}): {error_text[:200]}")
                        
                        # Nếu có lỗi nghiêm trọng, đợi và thử lại
                        if retry < MAX_RETRIES - 1:
                            await asyncio.sleep(2 * (retry + 1))
                            continue
            
            except (asyncio.TimeoutError, aiohttp.ClientError) as e:
                logger.error(f"Tin {news_idx}: Request error (lần {retry+1}/{MAX_RETRIES}): {str(e)}")
                if retry < MAX_RETRIES - 1:
                    await asyncio.sleep(2 * (retry + 1))
                    continue
    
    # Nếu tất cả các lần thử đều thất bại, trả về kết quả mặc định
    logger.warning(f"Tin {news_idx}: Tất cả các lần thử đều thất bại, sử dụng kết quả mặc định")
    default_result = get_default_result()
    with cache_lock:
        result_cache[cache_key] = default_result
    return default_result

# Trích xuất JSON từ văn bản
def extract_json(text):
    """Trích xuất JSON từ phản hồi văn bản."""
    try:
        # Trường hợp JSON hoàn chỉnh
        return json.loads(text)
    except:
        # Trường hợp JSON nằm trong văn bản
        try:
            import re
            json_match = re.search(r'({[\s\S]*})', text)
            if json_match:
                return json.loads(json_match.group(1))
        except:
            pass
        
        # Trả về None nếu không thể trích xuất
        return None

# Kết quả mặc định đã điều chỉnh cho phân tích cổ phiếu mục tiêu
def get_default_result():
    """Trả về kết quả mặc định"""
    return {
        "sentiment_label": "NEUTRAL",
        "sentiment_score": 0.0,
        "impact_score": 0.0,
        "relevance_score": 0.1,  # Mặc định là ít liên quan
        "target_mentioned": False,
        "direct_impact": False,
        "confidence": 0.5,
        "time_scores": {
            "short_term_score": 0.0,
            "medium_term_score": 0.0
        },
        "topic_weights": {
            "financial_performance": 0.2,
            "corporate_changes": 0.2,
            "market_conditions": 0.2,
            "regulatory_news": 0.2,
            "investor_sentiment": 0.2,
            "competition": 0.2,
            "technology": 0.2,
            "ai_sector": 0.2
        },
        "prominence": 5,
        "timeliness": 5
    }

# Chuyển đổi kết quả JSON phức tạp thành cấu trúc phẳng cho DataFrame
def flatten_json_for_dataframe(result):
    """Chuyển đổi kết quả JSON phức tạp thành cấu trúc phẳng cho DataFrame."""
    flat_result = {}
    
    # Xử lý các trường cơ bản
    for key, value in result.items():
        if key not in ['time_scores', 'topic_weights'] and not isinstance(value, dict):
            flat_result[key] = value
    
    # Xử lý time_scores
    if 'time_scores' in result:
        for key, value in result['time_scores'].items():
            flat_result[key] = value
    
    # Xử lý topic_weights
    if 'topic_weights' in result:
        for key, value in result['topic_weights'].items():
            flat_result[f"topic_{key}"] = value
    
    return flat_result

# Xử lý song song với nhiều API key
async def process_parallel(api_keys, news_list, start_idx):
    """Xử lý song song với số lượng tin tức bằng với số API key"""
    results = [None] * len(news_list)
    
    async with aiohttp.ClientSession() as session:
        tasks = []
        
        # Tạo các task cho mỗi tin tức với API key cố định
        for i, (news, api_key) in enumerate(zip(news_list, api_keys)):
            task = analyze_news(
                session,
                api_key,
                news['header'],
                news['content'],
                news['date'],
                start_idx + i
            )
            tasks.append((i, task))
        
        # Chờ và thu thập kết quả
        for i, task in tasks:
            try:
                result = await task
                results[i] = result
                
                # Hiển thị thông tin kết quả
                news = news_list[i]
                relevance = result.get('relevance_score', 0) * 100
                logger.info(f"Tin {start_idx + i}: {news['header'][:40]}... -> {result['sentiment_label']} ({result['sentiment_score']:.1f}) | Liên quan: {relevance:.1f}%")
            except Exception as e:
                logger.error(f"Lỗi khi xử lý tin {start_idx + i}: {str(e)}")
                results[i] = get_default_result()
    
    return results

# Kiểm tra và xóa cột trùng lặp
def clean_duplicate_columns(df):
    """Kiểm tra và xóa các cột kết quả phân tích đã có trong dữ liệu"""
    existing_columns = []
    for col in SENTIMENT_COLUMNS:
        if col in df.columns:
            existing_columns.append(col)
    
    if existing_columns:
        logger.info(f"Phát hiện {len(existing_columns)} cột phân tích đã tồn tại, đang xóa để tránh trùng lặp")
        df = df.drop(columns=existing_columns)
    
    return df

# Hàm main sử dụng asyncio
async def async_main():
    # Lấy danh sách API keys
    api_keys = get_api_keys()
    if not api_keys:
        logger.error("Lỗi: Không tìm thấy API key")
        sys.exit(1)
    
    num_keys = len(api_keys)
    logger.info(f"Sử dụng {num_keys} API key để xử lý song song")
    logger.info(f"Cổ phiếu mục tiêu: {TARGET_STOCK} ({STOCK_SYMBOL})")
    
    # Tạo thư mục lưu checkpoint
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    # Bắt đầu tính giờ
    start_time = time.time()
    
    try:
        # Đọc dữ liệu
        logger.info(f"Đọc file: {INPUT_FILE}")
        try:
            df = pd.read_csv(INPUT_FILE, encoding='utf-8')
        except UnicodeDecodeError:
            for encoding in ['latin1', 'cp1252', 'utf-8-sig']:
                try:
                    df = pd.read_csv(INPUT_FILE, encoding=encoding)
                    logger.info(f"Đọc file với encoding {encoding}")
                    break
                except:
                    continue
            else:
                logger.error("Không thể đọc file CSV với bất kỳ encoding nào")
                sys.exit(1)
        
        # Kiểm tra và xóa cột trùng lặp
        df = clean_duplicate_columns(df)
        
        # Kiểm tra cột
        required_cols = ['Created At', 'Header', 'Content']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            logger.error(f"Thiếu các cột: {', '.join(missing)}")
            sys.exit(1)
        
        total_records = len(df)
        logger.info(f"Đã đọc {total_records} dòng dữ liệu từ CSV")
        
        # Kiểm tra xem đã có file đầu ra chưa
        start_idx = 0
        output_df = None
        
        if os.path.exists(OUTPUT_FILE):
            logger.info(f"File đầu ra {OUTPUT_FILE} đã tồn tại, kiểm tra để tiếp tục từ checkpoint")
            try:
                output_df = pd.read_csv(OUTPUT_FILE, encoding='utf-8-sig')
                processed_count = len(output_df)
                
                # Kiểm tra xem có đúng là phần đã xử lý không
                if processed_count > 0 and 'sentiment_label' in output_df.columns:
                    logger.info(f"Đã xử lý {processed_count}/{total_records} tin tức trước đó")
                    start_idx = processed_count
                else:
                    logger.warning(f"File đầu ra không chứa kết quả phân tích đúng, bắt đầu lại từ đầu")
                    start_idx = 0
                    output_df = None
            except Exception as e:
                logger.warning(f"Lỗi khi đọc file đầu ra: {str(e)}, bắt đầu lại từ đầu")
                start_idx = 0
                output_df = None
        
        # Xác định số lượng tin tức còn lại cần xử lý
        remaining_records = total_records - start_idx
        
        if remaining_records <= 0:
            logger.info("Tất cả tin tức đã được xử lý, không cần làm gì thêm")
            sys.exit(0)
        
        logger.info(f"Còn {remaining_records} tin tức cần phân tích")
        
        # Chuẩn bị danh sách kết quả
        all_results = []
        
        # Xử lý từng lô với kích thước bằng số lượng API key
        num_batches = (remaining_records + num_keys - 1) // num_keys
        
        for batch_idx in range(num_batches):
            start_pos = start_idx + (batch_idx * num_keys)
            end_pos = min(start_pos + num_keys, total_records)
            current_batch_size = end_pos - start_pos
            
            # Lấy phần dữ liệu cho batch này
            batch_df = df.iloc[start_pos:end_pos]
            
            # Chuẩn bị dữ liệu và API keys cho batch này
            batch_data = []
            for _, row in batch_df.iterrows():
                header = row['Header'] if pd.notna(row['Header']) else ''
                content = row['Content'] if pd.notna(row['Content']) else ''
                date = row['Created At'] if pd.notna(row['Created At']) else ''
                
                batch_data.append({
                    'header': header,
                    'content': content,
                    'date': date
                })
            
            # Sử dụng đúng số lượng API key cần thiết
            batch_api_keys = api_keys[:current_batch_size]
            
            # Xử lý batch
            logger.info(f"Bắt đầu xử lý batch {batch_idx+1}/{num_batches}, tin tức {start_pos+1}-{end_pos}/{total_records}")
            batch_start_time = time.time()
            
            # Xử lý song song với mỗi API key xử lý một tin tức
            batch_results = await process_parallel(batch_api_keys, batch_data, start_pos)
            
            # Hiển thị thời gian xử lý
            batch_time = time.time() - batch_start_time
            avg_time_per_news = batch_time / current_batch_size if current_batch_size > 0 else 0
            logger.info(f"Đã xử lý batch {batch_idx+1}/{num_batches} trong {batch_time:.2f} giây " + 
                       f"(trung bình {avg_time_per_news:.2f} giây/tin)")
            
            # Thêm vào danh sách kết quả
            all_results.extend(batch_results)
            
            # Chuyển đổi kết quả thành DataFrame
            flat_results = [flatten_json_for_dataframe(r) for r in all_results]
            result_df = pd.DataFrame(flat_results)
            
            # Kết hợp với dữ liệu gốc
            batch_output_df = pd.concat([df.iloc[start_idx:start_idx+len(all_results)].reset_index(drop=True), result_df], axis=1)
            
            # Cập nhật file đầu ra
            if output_df is not None:
                # Nếu đã có sẵn dữ liệu, thêm vào cuối một cách an toàn
                # Chỉ lấy các cột gốc từ phần mới, không lấy các cột phân tích (tránh trùng lặp)
                original_columns = [col for col in df.columns if col not in SENTIMENT_COLUMNS]
                
                # Ghép dữ liệu mới
                new_data_rows = df.iloc[start_idx+len(output_df):start_idx+len(all_results)][original_columns].reset_index(drop=True)
                new_result_rows = result_df.iloc[len(output_df):].reset_index(drop=True)
                
                # Ghép dữ liệu mới với kết quả phân tích
                new_rows = pd.concat([new_data_rows, new_result_rows], axis=1)
                
                # Ghép với output_df hiện tại
                output_df = pd.concat([output_df, new_rows], ignore_index=True)
            else:
                output_df = batch_output_df
            
            # Lưu checkpoint
            if ((batch_idx + 1) % (CHECKPOINT_INTERVAL // num_keys) == 0) or (batch_idx == num_batches - 1):
                # Lưu checkpoint
                output_df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
                logger.info(f"Đã lưu checkpoint tại {start_idx+len(all_results)}/{total_records} tin tức")
            
            # Hiển thị tiến độ tổng thể
            progress = (start_idx + len(all_results)) / total_records * 100
            elapsed = time.time() - start_time
            speed = (start_idx + len(all_results) - start_idx) / elapsed if elapsed > 0 else 0
            remaining = (total_records - (start_idx + len(all_results))) / speed if speed > 0 else 0
            
            logger.info(f"Tiến độ tổng thể: {start_idx+len(all_results)}/{total_records} ({progress:.1f}%) | "
                       f"Tốc độ: {speed:.2f} tin/giây | "
                       f"Còn lại: {remaining/60:.1f} phút")
        
        # Lưu kết quả cuối cùng
        if output_df is not None:
            output_df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
            logger.info(f"Đã lưu kết quả cuối cùng vào {OUTPUT_FILE}")
            
            # Phân tích thống kê cảm xúc và mức độ liên quan
            if 'sentiment_label' in output_df.columns and 'relevance_score' in output_df.columns:
                # Thống kê cảm xúc tổng thể
                sentiment_counts = output_df['sentiment_label'].value_counts()
                logger.info("\nThống kê cảm xúc tổng thể:")
                for sentiment, count in sentiment_counts.items():
                    percentage = count/len(output_df)*100 if len(output_df) > 0 else 0
                    logger.info(f"{sentiment}: {count} ({percentage:.1f}%)")
                
                # Thống kê dựa trên mức độ liên quan
                relevant_news = output_df[output_df['relevance_score'] >= 0.5]
                relevant_count = len(relevant_news)
                logger.info(f"\nSố tin tức có liên quan đến {TARGET_STOCK} (relevance >= 0.5): {relevant_count}/{len(output_df)} ({relevant_count/len(output_df)*100:.1f}%)")
                
                if relevant_count > 0:
                    # Thống kê cảm xúc cho tin liên quan
                    relevant_sentiment = relevant_news['sentiment_label'].value_counts()
                    logger.info(f"\nThống kê cảm xúc cho tin liên quan đến {TARGET_STOCK}:")
                    for sentiment, count in relevant_sentiment.items():
                        percentage = count/relevant_count*100
                        logger.info(f"{sentiment}: {count} ({percentage:.1f}%)")
                    
                    # Tính điểm trung bình cho tin liên quan
                    logger.info(f"\nĐiểm trung bình cho tin liên quan đến {TARGET_STOCK}:")
                    for col in ['sentiment_score', 'impact_score', 'short_term_score', 'medium_term_score']:
                        if col in relevant_news.columns:
                            logger.info(f"{col}: {relevant_news[col].mean():.2f}")
                
                # Thống kê tin đề cập trực tiếp
                if 'target_mentioned' in output_df.columns:
                    mentioned_news = output_df[output_df['target_mentioned'] == True]
                    mentioned_count = len(mentioned_news)
                    logger.info(f"\nSố tin tức đề cập trực tiếp đến {TARGET_STOCK}: {mentioned_count}/{len(output_df)} ({mentioned_count/len(output_df)*100:.1f}%)")
            
            # Thống kê điểm trung bình tổng thể
            logger.info("\nĐiểm trung bình tổng thể:")
            for col in ['sentiment_score', 'impact_score', 'relevance_score', 'short_term_score', 'medium_term_score']:
                if col in output_df.columns:
                    logger.info(f"{col}: {output_df[col].mean():.2f}")
        
        # Tính thời gian
        total_time = time.time() - start_time
        logger.info(f"Hoàn thành toàn bộ quá trình trong {total_time/60:.2f} phút")
        
    except Exception as e:
        logger.exception(f"Lỗi không xử lý được: {str(e)}")
        sys.exit(1)

# Hàm chính
def main():
    # Thiết lập asyncio
    import platform
    
    if platform.system() == 'Windows':
        # Đảm bảo asyncio chạy đúng trên Windows
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    # Chạy main loop
    asyncio.run(async_main())

if __name__ == "__main__":
    main()