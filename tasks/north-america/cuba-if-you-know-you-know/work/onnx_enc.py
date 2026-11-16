import os, numpy as np, onnxruntime as ort
from tokenizers import Tokenizer
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "runs", "onnx")
def load(which="int8", threads=0, maxlen=320):
    so = ort.SessionOptions(); so.intra_op_num_threads = threads
    s = ort.InferenceSession(os.path.join(D, f"minilm_{which}.onnx"), so, providers=["CPUExecutionProvider"])
    t = Tokenizer.from_file(os.path.join(D, "tokenizer.json")); t.enable_truncation(maxlen); t.enable_padding()
    def enc(xs, bs=64):
        out = []
        for i in range(0, len(xs), bs):
            e = t.encode_batch(xs[i:i + bs])
            ids = np.array([x.ids for x in e], np.int64); m = np.array([x.attention_mask for x in e], np.int64)
            h = s.run(None, {"input_ids": ids, "attention_mask": m, "token_type_ids": np.zeros_like(ids)})[0]
            v = (h * m[..., None]).sum(1) / m.sum(1, keepdims=True)
            out.append(v / np.linalg.norm(v, axis=1, keepdims=True))
        return np.concatenate(out).astype(np.float32)
    return enc
