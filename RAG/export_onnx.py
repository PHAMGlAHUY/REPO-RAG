from pathlib import Path

import sentence_transformers
from sentence_transformers import SentenceTransformer, export_dynamic_quantized_onnx_model

OUT = Path(__file__).parent / 'data' / 'models' / 'bge-m3-onnx'
CONFIG = 'avx2'

model = SentenceTransformer('BAAI/bge-m3', backend='onnx')
model.save_pretrained(str(OUT))

model = SentenceTransformer(str(OUT), backend='onnx')
export_dynamic_quantized_onnx_model(model, CONFIG, str(OUT))
print("Xong: ", OUT)