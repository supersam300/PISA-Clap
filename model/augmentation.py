from clap import clap_model
from record_audio import record_audio
from openai import OpenAI
from liteLLM import complitions
ollama = openAI(base_url="http://localhost:11434",api_key="ollama")
model = "ollama/llama3.1"

result = clap_model()
lable = result[0][0]
confidence = result[0][1]
print(lable,confidence)


