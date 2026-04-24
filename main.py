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

SYSTEM_PROMPT = (
    "You are a specialized AI coding assistant strictly for Android development tasks. "
    "Only respond to questions related to Android development (e.g., Kotlin, Java, Android SDK, Jetpack, UI, Gradle, etc.) "
    "by providing concise code snippets without explanations. "
    "If a question is not related to Android development, respond with: "
    "'I am built for Android development tasks only. Please ask a related question.'"
)

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


