import logging
import os
import threading
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

from config.load_config import load_settings, _resolve

logger = logging.getLogger('embedding')
_cfg = load_settings()['embedding']

EMBEDDING_MODEL = _cfg['model']
EMBEDDING_BATCH_SIZE = _cfg.get('batch_size', 32)
QUERY_PREFIX = _cfg.get('query_prefix', '')
DOCUMENT_PREFIX = _cfg.get('document_prefix', '')
MAX_SEQ_LENGTH = _cfg.get('max_seq_length') or 512

if not (_cfg.get('onnx_dir') and _cfg.get('onnx_file')):
    raise ValueError("embedding config cần 'onnx_dir' và 'onnx_file'")
BACKEND = 'onnx'
ONNX_FILE = _cfg['onnx_file']
_ONNX_DIR = Path(_resolve(_cfg['onnx_dir']))
_ONNX_PATH = _ONNX_DIR / _cfg['onnx_file']
_TOKENIZER_PATH = _ONNX_DIR / 'tokenizer.json'

_want = _cfg.get('device', 'auto')
_has_cuda = 'CUDAExecutionProvider' in ort.get_available_providers()
EMBEDDING_DEVICE = 'cuda' if _has_cuda and _want in ('auto', 'cuda') else 'cpu'
_PROVIDERS = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if EMBEDDING_DEVICE == 'cuda' else ['CPUExecutionProvider']


class OnnxEmbedder:
    """Chạy ONNX (bge-m3) bằng onnxruntime: vector dense từ CLS, đã chuẩn hóa."""

    def __init__(self, onnx_path: Path, tokenizer_path: Path, max_len: int):
        so = ort.SessionOptions()
        so.intra_op_num_threads = os.cpu_count() or 4
        self.sess = ort.InferenceSession(str(onnx_path), so, providers=_PROVIDERS)
        self.tok = Tokenizer.from_file(str(tokenizer_path))
        self.tok.enable_truncation(max_length=max_len)  # không bật padding: tự pad theo batch bên dưới
        self.pad_id = self.tok.token_to_id('<pad>')
        if self.pad_id is None:
            self.pad_id = 1  # XLM-R
        self.input_names = {i.name for i in self.sess.get_inputs()}
        self.output_names = [o.name for o in self.sess.get_outputs()]
        self._dim: int | None = None

    @property
    def dim(self) -> int:
        if self._dim is None:
            self._dim = int(self.encode(['x']).shape[1])
        return self._dim

    def _run(self, batch: list[list[int]]) -> np.ndarray:
        n, width = len(batch), max(len(ids) for ids in batch)
        input_ids = np.full((n, width), self.pad_id, dtype=np.int64)
        mask = np.zeros((n, width), dtype=np.int64)
        for r, ids in enumerate(batch):
            input_ids[r, :len(ids)] = ids
            mask[r, :len(ids)] = 1
        feed = {k: v for k, v in {'input_ids': input_ids, 'attention_mask': mask}.items() if k in self.input_names}
        res = dict(zip(self.output_names, self.sess.run(None, feed)))
        if 'sentence_embedding' in res:
            return res['sentence_embedding']
        return res.get('last_hidden_state', next(iter(res.values())))[:, 0]  # CLS

    def encode(self, texts: list[str], batch_size: int = 32, normalize: bool = True) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)
        ids = [e.ids for e in self.tok.encode_batch(texts)]
        order = sorted(range(len(ids)), key=lambda i: len(ids[i]))  # gom theo số token, giảm padding
        parts = []
        for s in range(0, len(order), batch_size):
            v = self._run([ids[i] for i in order[s:s + batch_size]]).astype(np.float32, copy=False)
            if normalize:
                v = v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)
            parts.append(v)
        merged = np.concatenate(parts)
        result = np.empty_like(merged)
        result[order] = merged  # trả đúng thứ tự đầu vào
        return result


_model: OnnxEmbedder | None = None
_lock = threading.Lock()


def get_model() -> OnnxEmbedder:
    global _model
    if _model is None:
        with _lock:
            if _model is None:
                logger.info(f"Loading ONNX embedding model {EMBEDDING_MODEL} ({_ONNX_PATH.name}) on {EMBEDDING_DEVICE}")
                _model = OnnxEmbedder(_ONNX_PATH, _TOKENIZER_PATH, MAX_SEQ_LENGTH)
    return _model


def get_embedding_dim() -> int:
    return get_model().dim


def embed_texts(texts: list[str], is_query: bool = False) -> np.ndarray:
    """Mảng float32 shape (n, dim), đã chuẩn hóa."""
    prefix = QUERY_PREFIX if is_query else DOCUMENT_PREFIX
    if prefix:
        texts = [prefix + t for t in texts]
    return get_model().encode(texts, batch_size=EMBEDDING_BATCH_SIZE)


def embed_query(query: str) -> np.ndarray:
    return embed_texts([query], is_query=True)[0]


def warmup() -> None:
    embed_texts(['warmup'], is_query=True)