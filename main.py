import gradio as gr
import transformers
import torch

MODEL_ID = "haqishen/Llama-3-8B-Japanese-Instruct"
MAX_MEMORY_WORDS = 8192
MAX_NEW_TOKENS = 8192
TEMPERATURE = 0.6
TOP_P = 0.9

pipeline = transformers.pipeline(
    "text-generation",
    model=MODEL_ID,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device="cuda",
)

def chat(message, history):
    if not message:
        return None

    messages = [{"role": "user", "content": h[0]} for h in history]
    messages.append({"role": "user", "content": message})

    prompt = pipeline.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    terminators = [
        pipeline.tokenizer.eos_token_id,
        pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    outputs = pipeline(
        prompt,
        max_new_tokens=MAX_NEW_TOKENS,
        eos_token_id=terminators,
        do_sample=True,
        temperature=TEMPERATURE,
        top_p=TOP_P,
    )

    return outputs[0]["generated_text"][len(prompt):]

gr.ChatInterface(chat).launch(share=True)
