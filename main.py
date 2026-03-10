import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from huggingface_hub import AsyncInferenceClient
from dotenv import load_dotenv
load_dotenv()

app = FastAPI()

client = AsyncInferenceClient(
    api_key=os.environ.get("HF_TOKEN")
)

SYSTEM_PROMPT = ("You are a helpful AI coding assistant for the andriod development tasks. Don't give any messages just give code snippets.")

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    try:
        completion = await client.chat.completions.create(
            model="Qwen/Qwen3-Coder-Next:fastest",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": req.message}
            ]
        )

        return ChatResponse(
            response=completion.choices[0].message.content
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


