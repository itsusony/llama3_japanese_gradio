import gradio as gr
import transformers
import torch

MODEL_ID = "haqishen/Llama-3-8B-Japanese-Instruct"
MAX_MEMORY_WORDS = 8192

pipeline = transformers.pipeline(
    "text-generation",
    model=MODEL_ID,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device="cuda",
)

def chat(message, history, system_role, token, temperature, top_p):
    if not message:
        return history, ""

    messages = []
    if system_role:
        messages.append({"role": "system", "content": system_role})

    memory_len = 0
    for h in history:
        req = h[0]
        messages.append({"role": "user", "content": req})
        memory_len += len(req)
        if memory_len >= MAX_MEMORY_WORDS:
            break

    messages.append({"role": "user", "content": message})
    prompt = pipeline.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    terminators = [
        pipeline.tokenizer.eos_token_id,
        pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    outputs = pipeline(
        prompt,
        max_new_tokens=token,
        eos_token_id=terminators,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
    )

    response = outputs[0]["generated_text"][len(prompt):]
    history.append((message, response))
    return history, ""

with gr.Blocks() as webapp:
    with gr.Row():
        with gr.Column():
            system_role = gr.Textbox("", label="System Role", lines=5)
        with gr.Column():
            token = gr.Slider(128, 1024*100, 8192, step=128, label="Max New Token")
            top_p = gr.Slider(0, 1, 0.9, step=0.1, label="top_p")
            temperature = gr.Slider(0, 1, 0.6, step=0.1, label="temperature")

    chatbot = gr.Chatbot()
    msg = gr.Textbox(value="", label="Enter your message")
    clear = gr.Button("Clear")

    msg.submit(chat, [msg, chatbot, system_role, token, temperature, top_p], [chatbot, msg])
    clear.click(lambda: [], None, chatbot, queue=False)

webapp.launch(share=True)
