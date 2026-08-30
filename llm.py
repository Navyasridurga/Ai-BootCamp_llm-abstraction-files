import os

from langchain_nvidia_ai_endpoints import ChatNVIDIA
client=ChatNVIDIA(
    model="nvidia/nemotron-3-super-120b-a12b",
    api_key=os.getenv("NVIDIA_API_KEY"),
    temperature=0,
    top_p=1,
    max_completion_tokens=512
)
#"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"