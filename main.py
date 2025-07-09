from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

app = FastAPI()
chat_history = []

# Load model
model_name = "microsoft/phi-2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
model.to("cpu")

class ChatInput(BaseModel):
    message: str

def chat_with_phi2(prompt, history=[]):
    full_prompt = ""
    for user_msg, bot_reply in history:
        full_prompt += f"User: {user_msg}\nBot: {bot_reply}\n"
    full_prompt += f"User: {prompt}\nBot:"
    inputs = tokenizer(full_prompt, return_tensors="pt")
    output = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
        pad_token_id=tokenizer.eos_token_id
    )
    reply = tokenizer.decode(output[0], skip_special_tokens=True)
    reply = reply.split("Bot:")[-1].strip()
    history.append((prompt, reply))
    return reply

@app.post("/chat")
def get_reply(input: ChatInput):
    reply = chat_with_phi2(input.message, chat_history)
    return {"reply": reply}
