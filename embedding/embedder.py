import logging
from sentence_transformers import SentenceTransformer
from core.load_settings import load_settings

logger = logging.getLogger(__name__)
_cfg = load_settings()['embedding']
_model = SentenceTransformer(_cfg['model'], device=_cfg['device'])

def embed_passages(texts: list[str]):
    return _model.encode([f'passage: {t}' for t in texts],
                         batch_size = _cfg['batch_size'],
                         normalize_embeddings=True,
                         show_progress_bar=True)

def embed_query(text: str):
    return _model.encode(f'query: {text}', normalize_embeddings=True)