import torch, torch_directml, time, numpy as np
from transformers import AutoModel, AutoTokenizer

dml = torch_directml.device()
print("torch:", torch.__version__, "| DirectML:", torch_directml.device_name(0))

tok = AutoTokenizer.from_pretrained("microsoft/codebert-base")
code = "def get_user(id): cursor.execute(f\"SELECT * FROM users WHERE id={id}\")"
inputs = tok(code, return_tensors="pt", truncation=True, max_length=512)

print("Loading CPU model...")
m_cpu = AutoModel.from_pretrained("microsoft/codebert-base")
m_cpu.eval()
t = []
for _ in range(5):
    s = time.time()
    with torch.no_grad():
        m_cpu(**inputs, output_hidden_states=True)
    t.append(time.time()-s)
print(f"CPU: {np.mean(t)*1000:.0f}ms avg")

print("Loading DirectML model...")
m_dml = AutoModel.from_pretrained("microsoft/codebert-base").to(dml)
m_dml.eval()
inp_dml = {k: v.to(dml) for k,v in inputs.items()}
with torch.no_grad():
    m_dml(**inp_dml, output_hidden_states=True)
t2 = []
for _ in range(5):
    s = time.time()
    with torch.no_grad():
        m_dml(**inp_dml, output_hidden_states=True)
    t2.append(time.time()-s)
print(f"DirectML 780M: {np.mean(t2)*1000:.0f}ms avg")
print(f"Speedup: {np.mean(t)/np.mean(t2):.1f}x")
