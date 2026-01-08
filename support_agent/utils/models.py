from langchain_huggingface import HuggingFaceEndpoint
from langchain_huggingface import ChatHuggingFace
from dotenv import load_dotenv
import os

load_dotenv(".env", override=True)

llm= HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    temperature=0.0,
    max_new_tokens=1000,
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
)

llm_support = ChatHuggingFace(llm=llm)
