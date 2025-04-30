import os
import sys
import re
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import time
import platform
import logging
import hashlib
import psutil
import torch
import jieba
import numpy as np
import faiss
import yaml
import random
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from typing import List, Dict, Any, Optional, Generator, Tuple
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
from joblib import Parallel, delayed
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig,
    StoppingCriteria,
    StoppingCriteriaList,
    BitsAndBytesConfig
)
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


# ----- 模組1：系統配置  -----

class ConfigValidationError(Exception):
    """配置驗證異常基類"""

def safe_path(path: str) -> Path:
    """強化路徑安全處理 """
    return Path(str(path).encode('utf-8').decode('utf-8')).resolve()

def get_free_gpu_mem_gb(device_id=0) -> float:
    if not torch.cuda.is_available():
        return 1.0
    try:
        torch.cuda.empty_cache()
        free, _ = torch.cuda.mem_get_info(device_id)
        return max(free / (1024 ** 3), 0.5)
    except:
        return 1.0

@dataclass
class SystemConfig:
    """系統配置類 """
    generation_do_sample: bool = True  # 採樣模式開關
    generation_num_beams: int = 1      # beam search參數
    generation_min_tokens: int = 64
    generation_max_tokens: int = 1024
    generation_temperature: float = 0.62
    generation_top_p: float = 0.85
    max_context_length: int = 1536  # 與模型架構強相關
    cache_strategy: str = "lfu"     # 快取算法選擇
    bm25_weight: float = 0.55
    semantic_weight: float = 0.45
    retrieval_score_threshold: float = 0.72
    retrieval_top_k: int = 5

    # 模型路徑 (避免配置錯誤)
    model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    embed_model: str = r"C:\Users\User\Desktop\高爾夫 專題\.mypy_cache\models\final_model"
    
    # 系統級常數
    api_timeout: int = 30
    max_retries: int = 3

    faiss_cpu: Dict[str, Any] = field(default_factory=lambda: {
        "hnsw_ef_search": 128,
        "hnsw_ef_construction": 200,
        "batch_size": 64,
        "nlist": 512,
        "nprobe": 16,
        "max_threads": 12,
        "shard_size": 50000,
        "memory_safety_margin": 0.2,
        "min_text_length": 10
    })


    gpu_acceleration: bool = True
    gpu_device_id: int = 0
    max_concurrent_requests: int = 6
    generation_temperature: float = 0.7
    generation_top_p: float = 0.9
    generation_repetition_penalty: float = 1.315
    length_penalty: float = 1.05
    max_context_length: int = 1536
    log_level: int = logging.DEBUG
    gpu_memory_alloc: float = 0.75

    quantization: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": True,
        "model_quant_method": "bitsandbytes-nf4",  # 明確使用4bit
        "quant_level": "4bit",  
        "group_size": 64,    
        "compute_dtype": "fp16",
        "quantize_cache": True
    })

    @classmethod
    def from_yaml(cls, path: Path):
        config_path = safe_path(str(path))
        try:
            with open(config_path, 'r', encoding='utf-8-sig') as f:
                config_data = yaml.safe_load(f) or {}
                quant_cfg = config_data.get('quantization', {})
                config_data['quantization'] = quant_cfg
                return cls(**{k: v for k, v in config_data.items() if k in cls.__annotations__})
        except Exception as e:
            logging.critical(f"配置解析失敗: {str(e)}")
            raise ConfigValidationError("配置檔案格式錯誤") from e

    def __post_init__(self):
        if not hasattr(self, 'retrieval_top_k'):
            self.retrieval_top_k = min(8, int(os.cpu_count() * 0.8))  # 自動計算
        if abs((self.bm25_weight + self.semantic_weight) - 1.0) > 0.001:
            raise ConfigValidationError("檢索權重總和必須等於1.0")
        Path('cache').mkdir(exist_ok=True)
        Path('indices').mkdir(exist_ok=True)

# 需要在 SystemConfig 類定義之後
def select_quant_level(config: SystemConfig) -> str:
    if config.quantization.get('quant_level', 'auto') in ['4bit', '8bit']:
        return config.quantization['quant_level']
    try:
        free_mem = get_free_gpu_mem_gb(config.gpu_device_id)
        return '4bit' if free_mem < 6.0 else '8bit'
    except:
        return '4bit'


# ----- 模組2：資料載入器 (純文字版) -----
class StreamingDataLoader:
    """高效流式資料載入器 (純文字檔案)"""

    def __init__(self, config: SystemConfig):
        self.config = config
        self.logger = logging.getLogger('DataLoader')
        self.chunk_size = 5000

    def load_all(self, paths: List[Path]) -> Generator[List[str], None, None]:
        # 用Joblib多行程並行載入每個檔案
        results = Parallel(n_jobs=min(len(paths), self.config.faiss_cpu.get('max_threads', 12)))(
            delayed(self._load_file)(path) for path in paths if path.exists()
        )
        for chunks in results:
            for chunk in chunks:
                yield chunk

    def _load_file(self, path: Path) -> List[List[str]]:
        chunks = []
        try:
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read().strip()

            entries = [entry.strip() for entry in content.split('\n\n') if entry.strip()]
            chunk = []
            for entry in entries:
                if len(entry) >= self.config.faiss_cpu.get('min_text_length', 10):
                    chunk.append(entry)
                    if len(chunk) >= self.chunk_size:
                        chunks.append(chunk)
                        chunk = []
            if chunk:
                chunks.append(chunk)
            return chunks
        except Exception as e:
            self.logger.error(f"檔案 {path.name} 載入失敗: {str(e)}")
            return []

# ----- 模組3：檢索系統 (CPU) -----
class HybridRetriever:

    def __init__(self, data_paths: List[Path], config: SystemConfig):
        self.config = config
        self.logger = logging.getLogger('Retriever')
        self.data = []
        self.data_paths = data_paths 

        # 初始化 embedder（移到 __init__ 內）
        self.embedder = SentenceTransformer('final_model')
        self.embedder.max_seq_length = 256  # 設定最大序列長度

        self._validate_data_files(data_paths)
        self._load_data(data_paths)
        self._build_indices()
        self.logger.info(f"檢索系統就緒 | 總資料量: {len(self.data):,}")

    def _validate_data_files(self, paths: List[Path]):
        missing_files = [p for p in paths if not p.exists()]
        if missing_files:
            raise FileNotFoundError(f"缺少資料檔案: {', '.join(f.name for f in missing_files)}")

    def _load_data(self, paths: List[Path]):
        try:
            loader = StreamingDataLoader(self.config)
            total_count = 0
            for chunk in loader.load_all(paths):
                self.data.extend(chunk)
                total_count += len(chunk)
                if total_count % 10000 == 0:
                    self.logger.debug(f"已載入資料: {total_count:,}條")

            if not self.data:
                self.logger.critical("❌ 資料載入失敗：未找到有效資料條目")
                raise ValueError("所有資料檔案均無有效內容")

            self.logger.info(f"資料載入完成 | 總條目數: {total_count:,}")
        except Exception as e:
            self.logger.error(f"資料載入異常: {str(e)}")
            raise

    def _build_indices(self):
        from tqdm import tqdm
        total_docs = len(self.data)
        batch_size = 64  # <--- 定義批次大小


        for path in self.data_paths:
            if not path.exists():
                self.logger.warning(f"跳過不存在的資料檔案: {path}")
                continue
            source_name = path.stem.split('_')[0]  # 取得資料來源名稱
            faiss_base = self.config.faiss_cpu.get('base_params', {})
            source_params = self.config.faiss_cpu.get(source_name, {})
            nlist = source_params.get('nlist', faiss_base.get('nlist', 512))
            nprobe = source_params.get('nprobe', faiss_base.get('nprobe', 16))
            self.logger.info(f"正在為 {source_name} 建立索引 | nlist={nlist}, nprobe={nprobe}")


            zh_docs = [d for d in self.data if any('\u4e00' <= c <= '\u9fff' for c in d)]
            en_docs = [d for d in self.data if not any('\u4e00' <= c <= '\u9fff' for c in d)]

        if zh_docs:
            self.logger.info("建立中文BM25索引...")
            zh_tokenized = [list(jieba.cut(doc)) for doc in zh_docs]
            self.bm25_zh = BM25Okapi(zh_tokenized)
        else:
            self.logger.warning("未偵測到中文資料，跳過中文索引建構")
            self.bm25_zh = None

        if en_docs:
            self.logger.info("建立英文BM25索引...")
            en_tokenized = [doc.split() for doc in en_docs]
            self.bm25_en = BM25Okapi(en_tokenized)
        else:
            self.logger.warning("未偵測到英文資料，跳過英文索引建構")
            self.bm25_en = None

        if not self.data:
            raise ValueError("無有效資料可供建立語義索引")

        self.logger.info("建立FAISS語義索引...")
        self.embedder = SentenceTransformer('final_model')

        embeddings = []
        for i in tqdm(range(0, total_docs, batch_size), desc="Embedding"):
            batch = self.data[i:i+batch_size]
            emb = self.embedder.encode(batch, convert_to_numpy=True)
            embeddings.append(emb)
        embeddings = np.concatenate(embeddings)

        
        # 使用HNSW替代IVF提升檢索速度
        dim = embeddings.shape[1]
        index = faiss.IndexHNSWFlat(dim, 32)
        index.add(embeddings.astype(np.float32))
        self.index = index  # 保持使用HNSW索引

        self.logger.info("FAISS HNSW索引建立完成")

        # 在 HNSW 索引建構後添加參數檢查
        assert self.index.is_trained, "索引未正確訓練"
        self.logger.debug(f"索引類型: {type(self.index).__name__}")

    def retrieve(self, query: str) -> List[Tuple[float, str]]:
        is_chinese = any('\u4e00' <= c <= '\u9fff' for c in query)

        if is_chinese and self.bm25_zh:
            tokens = list(jieba.cut(query))
            bm25_scores = self.bm25_zh.get_scores(tokens)
            bm25_scores = np.array(bm25_scores)
        elif self.bm25_en:
            tokens = query.split()
            bm25_scores = self.bm25_en.get_scores(tokens)
            bm25_scores = np.array(bm25_scores)
        else:
            bm25_scores = np.zeros(len(self.data))

        query_embed = self.embedder.encode([query])
        distances, semantic_indices = self.index.search(query_embed.astype(np.float32), 100)
        distances = distances[0]
        semantic_indices = semantic_indices[0]

        combined = []
        for rank, idx in enumerate(semantic_indices):
            if idx == -1 or idx >= len(self.data):
                continue
            bm25_score = bm25_scores[idx] if idx < len(bm25_scores) else 0
            semantic_score = distances[rank] if distances[rank] > 0 else 0
            combined_score = (self.config.bm25_weight * bm25_score +
                              self.config.semantic_weight * semantic_score)
            combined.append((combined_score, self.data[idx]))

        filtered_sorted = sorted(
            [item for item in combined if item[0] > self.config.retrieval_score_threshold],
            key=lambda x: x[0], reverse=True
        )
        return filtered_sorted[:self.config.retrieval_top_k]

# ----- 模組4：生成系統 (GPU) -----

class GolfResponseGenerator:
    def __init__(self, config: SystemConfig):
        self.config = config
        self.logger = logging.getLogger('ResponseGenerator')
        self.device = self._init_device()
        self.quant_level = self._select_quant_level()
        self.tokenizer, self.model = self._load_model()
        self.stopping_criteria = StoppingCriteriaList([self._StopOnKeywords(self.tokenizer)])
        self.logger.info(f"生成系統準備就緒 | 裝置:{self.device} | 量化: {self.quant_level}")

    def _init_device(self) -> str:
        if torch.cuda.is_available() and self.config.gpu_acceleration:
            torch.backends.cuda.matmul.allow_tf32 = True
            return f"cuda:{self.config.gpu_device_id}"
        return "cpu"
    
    def _select_quant_level(self) -> str:
        """針對8GB顯示卡最佳化的量化選擇"""
        try:
            # 根據可用顯示記憶體動態調整
            free_mem = get_free_gpu_mem_gb(self.config.gpu_device_id)
            if free_mem > 5.5:  # 保留2.5GB給系統
                return '8bit' if free_mem > 6.5 else '4bit' 
            return '4bit'
        except:
            return '4bit'  # 安全回退

    def _log_gpu_stats(self):
        """記錄 GPU 狀態但不中斷執行"""
        if torch.cuda.is_available():
            try:
                total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                used = total - (get_free_gpu_mem_gb() or 0)
                self.logger.debug(
                    f"GPU 狀態 | 已用: {used:.1f}GB / 總共: {total:.1f}GB "
                    f"| 使用率: {(used/total)*100:.1f}%"
                )
            except Exception as e:
                self.logger.warning(f"GPU 監測異常: {str(e)}")

    def _load_model(self) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
        try:
            tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Llama-8B", use_fast=True)
            model = AutoModelForCausalLM.from_pretrained(
                "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
                device_map="auto",  
                torch_dtype=torch.float16,  # 使用FP16以節省顯示記憶體
                low_cpu_mem_usage=True
            )
            model.eval()
            return tokenizer, model
        except RuntimeError as e:
            if 'CUDA out of memory' in str(e):
                self.logger.warning("啟動緊急顯示記憶體回收模式...")
                torch.cuda.empty_cache()
                return self._load_model()  # 重試載入
            raise

     

    def _dynamic_generation_config(self, prompt: str) -> GenerationConfig:
        """根據輸入動態調整生成參數"""
        prompt_length = len(prompt.split())
        dynamic_max = min(
            max(prompt_length * 2, self.config.generation_min_tokens),
            self.config.generation_max_tokens
        )
        
        return GenerationConfig(
            max_new_tokens=dynamic_max,
            min_new_tokens=self.config.generation_min_tokens,
            temperature=self.config.generation_temperature,
            top_p=self.config.generation_top_p,
            repetition_penalty=self.config.generation_repetition_penalty,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            num_beams=1,  # 避免與 length_penalty 衝突
            length_penalty=1.0  # 暫時停用複雜設定
        )

    def generate(self, query: str, context: List[str]) -> str:
        torch.cuda.empty_cache()  # 每次生成前強制清理
        try:
            self._log_gpu_stats()
            
            prompt = self._build_prompt(query, context)
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                max_length=self.config.max_context_length,
                truncation=True
            ).to(self.device)

            generation_config = self._dynamic_generation_config(prompt)
            
            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    generation_config=generation_config,
                    stopping_criteria=self.stopping_criteria
                )

            generated_ids = outputs[0][inputs.input_ids.shape[1]:]
            response = self._format_response(generated_ids)
            
            return response
            
        except RuntimeError as e:
            if 'CUDA' in str(e):
                self._emergency_memory_clean() 
                return self._fallback_response("資源釋放中，請簡化問題後重試")
            
    def _emergency_memory_clean(self):
        """三階段緊急清理"""
        for _ in range(3):
            torch.cuda.empty_cache()
            time.sleep(0.5)
        gc.collect()

    def _cleanup_memory(self):
        """加強型記憶體清理"""
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                with torch.cuda.device(self.device):
                    torch.cuda.memory_summary(device=self.device, abbreviated=True)
            except Exception as e:
                self.logger.warning(f"記憶體清理異常: {str(e)}")

    TYPE_KEYWORDS = {
        '規則條款': ['規則', '違規', '罰桿', '判例', '條款'],
        '技術解析': ['調整', '技巧', '改善', '如何打', '設定'],
        '裝備諮詢': ['球桿', '裝備', '規格', '認證']
    }

    GENERATION_TEMPLATES = {
        '規則條款': {
            'structure': [
                "1️⃣ [規則編號] 核心條款精要",
                "2️⃣ 近期賽事應用案例",
                "3️⃣ 常見執行誤差提醒"
            ],
            'examples': [
                "根據2024年USGA第{隨機數字}條修正案...",
                "如上週PGA巡迴賽中{選手}的情況..."
            ],
            'closings': [
                "\n\n詳細規則變動請查閱最新USGA手冊",
                "\n\n實際判罰需由現場裁判決定"
            ]
        },
        '技術解析': {
            'structure': [
                "1️⃣ 生物力學原理分析",
                "2️⃣ 動作要領分解",
                "3️⃣ 不同場地適用技巧"
            ],
            'examples': [
                "根據TrackMan數據統計顯示...",
                "建議將桿面角調整{數字}度..."
            ],
            'closings': [
                "\n\n建議配合專業教練指導練習",
                "\n\n可透過慢動作錄影檢視動作細節"
            ]
        },
        '裝備諮詢': {
            'structure': [
                "1️⃣ USGA認證標準",
                "2️⃣ 主流品牌型號比較",
                "3️⃣ 規格調整建議"
            ],
            'examples': [
                "根據2023年裝備測試報告...",
                "{品牌}最新款在{參數}方面..."
            ],
            'closings': [
                "\n\n購買前建議實際試打體驗",
                "\n\n規格需配合個人體型調整"
            ]
        },
        'default': {
            'structure': [
                "1️⃣ 核心概念解析",
                "2️⃣ 常見問題解答",
                "3️⃣ 進階練習建議"
            ],
            'examples': [
                "根據TPGA教練手冊建議...",
                "多數職業選手採用{技術}的原因是..."
            ],
            'closings': [
                "\n\n更多疑問歡迎進一步諮詢",
                "\n\n實際情況可能因場地條件有所不同"
            ]
        }
    }

    def _detect_question_type(self, query: str) -> str:
        """智能問題分類"""
        lower_query = query.lower()
        for q_type, keywords in self.TYPE_KEYWORDS.items():
            if any(kw in lower_query for kw in keywords):
                return q_type
        return 'default'

    def _build_prompt(self, query: str, context: List[str]) -> str:
        """動態多樣化模板"""
        q_type = self._detect_question_type(query)
        template = self.GENERATION_TEMPLATES[q_type]
    
        # 限制上下文長度
        context_str = "\n".join(
            f"[[{i+1}]] {text[:80]}"  # 從100縮減到80字符
            for i, text in enumerate(context[:1])  # 從2條減到1條
        ) if context else "無相關資料"

        # 簡化模板結構
        return f"""請用繁體中文回答高爾夫問題：
    [問題類型] {q_type}
    [問題] {query[:120]}  # 限制問題長度
    [參考] {context_str}
    [回答要求] 請簡要說明重點"""

    def _dynamic_generation_config(self, prompt: str) -> GenerationConfig:
        """智能參數調整"""
        q_type = self._detect_question_type(prompt)
    
        # 根據問題類型調整參數
        type_params = {
            '規則條款': {'temp': 0.4, 'top_p': 0.78, 'max_tokens': 380},
            '技術解析': {'temp': 0.65, 'top_p': 0.88, 'max_tokens': 480},
            '裝備諮詢': {'temp': 0.55, 'top_p': 0.82, 'max_tokens': 480},
            'default': {'temp': 0.5, 'top_p': 0.85, 'max_tokens': 400}
        }[q_type]
    
        return GenerationConfig(
            max_new_tokens=type_params['max_tokens'], 
            temperature=0.5,  # 固定值簡化
            top_p=0.85,
            do_sample=True
        )


    def _format_response(self, output_ids: torch.Tensor) -> str:
        """自然化格式處理"""
        text = self.tokenizer.decode(output_ids, skip_special_tokens=True)
    
        # 隨機選擇不同的項目符號樣式
        symbol_sets = [
            ['❶', '❷', '❸'],  # 圓圈數字
            ['▸', '▸', '▸'],  # 三角形
            ['•', '•', '•'],   # 圓點
            ['→', '→', '→'],   # 箭頭
            ['✓', '✓', '✓']    # 勾選
        ]
        symbols = random.choice(symbol_sets)
    
        # 動態替換編號
        for i in range(1, 4):
            text = text.replace(f"{i}.", symbols[i-1])
    
        # 清理多餘空行
        text = re.sub(r'\n{3,}', '\n\n', text)
    
        # 添加類型對應結尾語
        q_type = self._detect_question_type(text)  # 從生成內容反推類型
        closing = random.choice(self.GENERATION_TEMPLATES[q_type]['closings'])
    
        return text + closing

    def _fallback_response(self, reason="") -> str:
        fallbacks = [
            "您詢問的規則需要更多現場資訊，建議諮詢PGA認證教練",
            "當前問題需結合比賽情境判斷，請提供更多細節",
            "此規則可能需要專業裁判現場判定"
        ]
        msg = random.choice(fallbacks)
        return f"{reason}\n{msg}" if reason else msg

    class _StopOnKeywords(StoppingCriteria):
        def __init__(self, tokenizer):
            super().__init__()
            self.stop_phrases = ["[回答結束]", "以上是", "總結來說"]
            self.tokenizer = tokenizer

        def __call__(self, input_ids, scores, **kwargs):
            decoded = self.tokenizer.decode(input_ids[0][-25:])
            return any(phrase in decoded for phrase in self.stop_phrases)

# ----- 模組5：主系統整合 -----

class MemoryCacheManager:
    """記憶體快取管理 (自訂LRU)"""
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.cache: Dict[str, str] = {}
        self._access_order: List[str] = []
        self.hit_count = 0
        self.miss_count = 0

    def get(self, key: str) -> Optional[str]:
        value = self.cache.get(key)
        if value is not None:
            self.hit_count += 1
            self._update_access_order(key)
        else:
            self.miss_count += 1
        return value

    def set(self, key: str, value: str):
        if key in self.cache:
            self.cache[key] = value
            self._update_access_order(key)
            return
        if len(self.cache) >= self.max_size:
            lru_key = self._access_order.pop(0)
            del self.cache[lru_key]
        self.cache[key] = value
        self._update_access_order(key)

    def _update_access_order(self, key: str):
        if key in self._access_order:
            self._access_order.remove(key)
        self._access_order.append(key)

    @property
    def stats(self) -> dict:
        total = self.hit_count + self.miss_count
        return {
            'hit_rate': self.hit_count / total if total > 0 else 0,
            'total_queries': total,
            'current_size': len(self.cache),
            'lru_queue': self._access_order[-5:]
        }

class GolfExpertSystem:
    """高爾夫規則問答系統核心類"""

    def __init__(self, data_paths: List[Path], config_path: Optional[Path] = None):
        self.data_files = data_paths
        self.logger = logging.getLogger('GolfSystem')
        self.config = self._load_config(config_path)
        self._validate_data_files()
        self.retriever = HybridRetriever(data_paths, self.config)
        self.generator = GolfResponseGenerator(self.config)
        self.cache = MemoryCacheManager()
        self._initialize_hardware()
        # 生成請求使用ThreadPoolExecutor，資料載入用Joblib已最佳化
        self.executor = ThreadPoolExecutor(
            max_workers=self.config.max_concurrent_requests,
            thread_name_prefix='golf_worker',
            initializer=self._init_worker
        )
        self.logger.info("LLM系統啟動完成")

    def _load_config(self, config_path: Optional[Path]) -> SystemConfig:
        try:
            final_path = config_path or Path("config/system_config.yaml")
            return SystemConfig.from_yaml(final_path)
        except Exception as e:
            self.logger.critical(f"配置載入失敗: {str(e)}")
            raise RuntimeError("系統配置初始化失敗") from e

    def _validate_data_files(self):
        missing_files = [f for f in self.data_files if not f.exists()]
        if missing_files:
            missing_names = ", ".join(f.name for f in missing_files)
            self.logger.critical(f"缺少必要資料檔案: {missing_names}")
            sys.exit(1)

    def _initialize_hardware(self):
        try:
            if torch.cuda.is_available() and self.config.gpu_acceleration:
                mem_ratio = min(0.9, self.config.gpu_memory_alloc)
                torch.cuda.set_per_process_memory_fraction(mem_ratio)
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.benchmark = True
                self.logger.info(f"GPU加速已啟用 | 記憶體分配: {mem_ratio*100:.1f}%")
            else:
                self.logger.warning("GPU加速未啟用，使用CPU模式")
        except Exception as e:
            self.logger.critical(f"硬體初始化失敗: {str(e)}", exc_info=True)
            sys.exit(1)

    def _init_worker(self):
        torch.set_num_threads(1)
        logging.getLogger().setLevel(self.config.log_level)

    def _generate_cache_key(self, text: str) -> str:
        return hashlib.sha256(text.encode('utf-8')).hexdigest()

    def query(self, question: str) -> str:
        cache_key = self._generate_cache_key(question)
        cached = self.cache.get(cache_key)
        if cached:
            self.logger.debug(f"快取命中: {cache_key[:8]}")
            return cached

        future_retrieve = self.executor.submit(self.retriever.retrieve, question)
        context = future_retrieve.result()
        context_texts = [text for _, text in context]

        future_generate = self.executor.submit(self.generator.generate, question, context_texts)
        response = future_generate.result()

        self.cache.set(cache_key, response)
        return response

    def run(self):
        """簡易互動式介面"""
        print("歡迎使用高爾夫問答系統，輸入 'exit' 離開。")
        while True:
            question = input("請輸入您的問題：").strip()
            if question.lower() == 'exit':
                print("系統結束")
                break
            answer = self.query(question)
            print("回答：\n", answer)

# ----- 輔助工具 -----
class TermNormalizer:

    def __init__(self, config: SystemConfig):
        if not hasattr(config, 'safety_filters'):
            raise ValueError("安全過濾配置缺失")
        self.safety_patterns = config.safety_filters
        self._mapping = {
            '球员': '球員',
            '发球台': '開球臺',
            # ... 其他對應
        }

    _mapping = {
        '球员': '球員', '发球台': '開球臺',
        '杆': '桿', '台': '臺', '标准': '標準',
        'birdie': '小鳥球', 'eagle': '老鷹球',
        'bunker': '沙坑', 'par': '標準桿'
    }

    def normalize(self, text: str) -> str:
        # 安全過濾優先執行
        for pattern in self.safety_patterns:
            text = re.sub(pattern, '[內容已過濾]', text, flags=re.IGNORECASE)  # 新增flag忽略大小寫
    
        # 術語標準化處理
        for cn, tw in self._mapping.items():
            text = text.replace(cn, tw)
    
        # 清理多餘空白
        text = ' '.join(text.split())
    
        return text

    def apply_safety_filters(self, text):
        for pattern in config.safety_filters:
            text = re.sub(pattern, '[過濾內容]', text)
        return text

class SystemMonitor:
    def __init__(self):
        self.start_time = time.time()

    @property
    def uptime(self) -> str:
        sec = time.time() - self.start_time
        return f"{int(sec // 3600)}h {int(sec % 3600 // 60)}m"

    def resource_usage(self) -> dict:
        mem = psutil.Process().memory_info().rss // 1024**2
        return {
            'memory_mb': mem,
            'cpu_percent': psutil.cpu_percent(),
            'uptime': self.uptime
        }

# ----- 系統初始化 -----
class SystemInitializer:
    def __init__(self):
        self.emergency_log_fd = None
        self._setup_emergency_log()
        self._configure_system_encoding()
        self._create_essential_directories()
        self._initialize_logging_system()

    def _setup_emergency_log(self):
        try:
            self.emergency_log_fd = os.open("emergency.log", os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
            os.write(self.emergency_log_fd, "\n=== 系統啟動 ===\n".encode('utf-8'))
        except Exception as e:
            print(f"嚴重錯誤: 無法建立日誌檔案 {str(e)}", file=sys.stderr)
            os._exit(1)

    def _log_emergency_message(self, message: str):
        if self.emergency_log_fd:
            os.write(self.emergency_log_fd, f"[緊急事件] {message}\n".encode('utf-8', 'replace'))

    def _configure_system_encoding(self):
        try:
            if sys.platform == "win32":
                os.environ["PYTHONUTF8"] = "1"
                os.environ["PYTHONLEGACYWINDOWSSTDIO"] = "utf-8"
                sys.getfilesystemencoding = lambda: 'utf-8'
            sys.stdin.reconfigure(errors='replace')
            sys.stdout.reconfigure(errors='replace')
            sys.stderr.reconfigure(errors='replace')
        except Exception as e:
            self._log_emergency_message(f"編碼設定失敗: {str(e)}")
            os._exit(1)

    def _create_essential_directories(self):
        required_dirs = {
            "logs": 0o755,
            "cache": 0o700,
            "models": 0o555,
            "data": 0o755
        }
        for dir_name, mode in required_dirs.items():
            try:
                dir_path = Path(dir_name)
                if not dir_path.exists():
                    dir_path.mkdir(parents=True, exist_ok=True)
                    os.chmod(dir_path, mode)
                    self._log_emergency_message(f"已建立目錄: {dir_path}")
            except Exception as e:
                self._log_emergency_message(f"目錄建立失敗 {dir_path}: {str(e)}")
                os._exit(1)

    def _initialize_logging_system(self):
        try:
            log_format = '%(asctime)s.%(msecs)03d | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s'
            date_format = '%Y-%m-%d %H:%M:%S'

            file_handler = logging.FileHandler('logs/system.log', encoding='utf-8', errors='backslashreplace')
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(logging.Formatter(log_format, date_format))

            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            console_handler.setFormatter(logging.Formatter(log_format, date_format))

            logging.basicConfig(level=logging.DEBUG, handlers=[file_handler, console_handler])

            logging.getLogger("urllib3").setLevel(logging.WARNING)
            logging.getLogger("faiss").setLevel(logging.WARNING)

            logging.getLogger("Main").info("日誌系統初始化完成")
        except Exception as e:
            self._log_emergency_message(f"日誌初始化失敗: {str(e)}")
            os._exit(1)
        finally:
            if self.emergency_log_fd:
                os.close(self.emergency_log_fd)

# ----- 主程式入口 -----
if __name__ == "__main__":
    SystemInitializer()
    
    data_files = [
        Path("data/garoc.txt"),
        Path("data/golfalot.txt"),
        Path("data/golfdigest.txt"),
        Path("data/golfshop_tw.txt"),
        Path("data/pgatour.txt"),
        Path("data/usga.txt"),
        Path("data/wikipedia_golf.txt")
    ]

    try:
        expert_system = GolfExpertSystem(data_paths=data_files)
        expert_system.run()
    except Exception as e:
        logging.critical(f"系統啟動失敗: {str(e)}", exc_info=True)
        sys.exit(1)