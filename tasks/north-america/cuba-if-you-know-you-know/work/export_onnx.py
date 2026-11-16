# export MiniLM to ONNX (fp32) and make an int8 dynamic-quantised copy
import os, torch
from transformers import AutoTokenizer, AutoModel
from onnxruntime.quantization import quantize_dynamic, QuantType
name = "sentence-transformers/all-MiniLM-L6-v2"
out = os.path.join(os.path.dirname(__file__), "..", "runs", "onnx")
tok = AutoTokenizer.from_pretrained(name); model = AutoModel.from_pretrained(name).eval()
class Wrap(torch.nn.Module):
    def __init__(self, m): super().__init__(); self.m = m
    def forward(self, input_ids, attention_mask, token_type_ids):
        return self.m(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids).last_hidden_state
x = tok(["hello world"], return_tensors="pt")
torch.onnx.export(Wrap(model), (x["input_ids"], x["attention_mask"], x["token_type_ids"]), os.path.join(out, "minilm_fp32.onnx"),
                  input_names=["input_ids", "attention_mask", "token_type_ids"], output_names=["last_hidden_state"],
                  dynamic_axes={k: {0: "b", 1: "s"} for k in ["input_ids", "attention_mask", "token_type_ids", "last_hidden_state"]},
                  opset_version=17, dynamo=False)
quantize_dynamic(os.path.join(out, "minilm_fp32.onnx"), os.path.join(out, "minilm_int8.onnx"), weight_type=QuantType.QInt8)
tok.backend_tokenizer.save(os.path.join(out, "tokenizer.json"))
for f in os.listdir(out): print(f, os.path.getsize(os.path.join(out, f)) / 1e6)
