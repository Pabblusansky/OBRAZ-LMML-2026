# Egypt – Mereu valid, fără scuze (Ușor)

Istoricii din Cairo au trecut prin OCR mii de cărți vechi de egiptologie și au extras fișe de evidență, câte una pentru fiecare personaj sau monument, sub formă de text simplu `key: value`. Catalogul digital acceptă doar JSON strict valid. Folosește oracolul (Qwen3-1.7B local), nu propria mână. 99 înseamnă eșec.

Obiectiv: `main.py` citește `test-input.txt` din propriul director și scrie `test-output.txt` alături (linia N = JSON pentru linia N). Neinteractiv, offline, GPU, limită 60 s.
FLAG: `SIGMOID_<WORD>`, eliberat doar la 100/100.

Reguli: câmpuri `key: value` separate prin `|`; trim; zgomot (page 22, Chapter IX, - 45 -, [torn], (margin), ###, Fig) eliminat; doar `cuvânt: valoare` e câmp. Gol → null; true/false → bool; întreg → număr; altfel string (escape `"` și `\`). Doar primul `:` separă.
Model: `huggingface-cli download Qwen/Qwen3-1.7B-GGUF Qwen3-1.7B-Q8_0.gguf --local-dir models`; `llama-cpp-python==0.3.35` cu CUDA, `n_gpu_layers=-1`.
