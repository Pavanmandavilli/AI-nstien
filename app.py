import os
import json
import base64
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from openai import OpenAI
import uvicorn
from dotenv import load_dotenv

load_dotenv()
app = FastAPI()


client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.getenv("HF_API_KEY")
)


@app.websocket("/ws/interview")
async def interview_handler(websocket: WebSocket):
    await websocket.accept()

    try:
        while True:
            data = await websocket.receive_text()
            payload = json.loads(data)

            transcript = payload.get("transcript", "")
            base64_image = payload.get("screenshot", "")
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are an expert technical interviewer. The user is in a live interview. "
                        "Look at the code/screen provided and the interviewer's question. "
                        "Provide the EXACT technical answer or code snippet required. "
                        "Be concise but complete. Do not say 'Here is the answer', just give the answer."
                    )
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"Interviewer asked: {transcript}"},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]
                }
            ]

            try:
                # 3. CALL DEEPSEEK-R1
                completion = client.chat.completions.create(
                    model="deepseek-ai/DeepSeek-R1:fastest",
                    messages=messages,
                )

                # Get the exact answer
                full_answer = completion.choices[0].message.content

                # Send back to the hidden Electron window
                await websocket.send_json({"hint": full_answer})

            except Exception as ai_err:
                print(f"AI Error: {ai_err}")
                await websocket.send_json({"hint": "Error reaching DeepSeek..."})

    except WebSocketDisconnect:
        print("Line disconnected.")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)