from gradio.interface import Interface
from transformers import BertTokenizerFast, GPT2LMHeadModel
import torch
import gradio as gr


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
model_checkpoint = r"delex_end2end_output_with_none_repeat_action"
tokenizer = BertTokenizerFast.from_pretrained(model_checkpoint, padding_side="left")
model = GPT2LMHeadModel.from_pretrained(model_checkpoint)
model.config.eos_token_id = 102  # 102  # 21131
model.to(device)


def greet(input_text):

    tokens_tensor = tokenizer(
        input_text,
        add_special_tokens=False,
        truncation=True,
        return_tensors="pt",
    )["input_ids"].to(device)

    beam_output = model.generate(
        tokens_tensor,
        num_beams=1,
        num_return_sequences=1,
        max_length=32,
        early_stopping=True,
        pad_token_id=21131,
    )
    decode_text = tokenizer.decode(beam_output[0])
    print(decode_text)

    return ""


iface = gr.Interface(fn=greet, inputs=["text"], outputs=["text"])
iface.launch()