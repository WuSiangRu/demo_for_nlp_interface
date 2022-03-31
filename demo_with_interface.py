import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, BertTokenizerFast
import sys, os
from evaluate_multiwoz_zhtw import MultiWozDB
from lexicalize_response import lexicalize_restaurant, lexicalize_attraction, lexicalize_hotel, lexicalize_train
from utils.simpletod import get_belief, clean_belief, convert_belief, get_response_new
import pprint
import logging
import time
from gradio.interface import Interface
import gradio as gr



def get_belief_new_dbsearch(sent):
    if "<|belief|>" in sent:
        tmp = sent.strip(" ").split("<|belief|>")[-1].split("<|endofbelief|>")[0]
    else:
        return []
    tmp = tmp.strip(" .,")
    # assert tmp.endswith('<endofbelief>')
    tmp = tmp.replace("<|endofbelief|>", "")
    tmp = tmp.replace("[CLS]", "")
    tmp = tmp.replace("[SEP]", "")
    belief = tmp.split(",")
    new_belief = []
    for bs in belief:
        # bs = bs.strip(' .,')
        bs = "".join(bs.split())
        if bs not in new_belief:
            new_belief.append(bs)
    return new_belief


def get_response_new(sent):
    if "<|response|>" in sent:
        return (sent.split("<|response|>")[-1].split("<|endofresponse|>")[0])
    else:
        return ""


def get_turn_domain(beliefs, q):
    for k in beliefs.keys():
        if k not in q:
            q.append(k)
            return k
    return q[-1]

def get_action(sent):
    action_text = ""
    if "<|action|>" in sent:
        action_text = sent.split("<|action|>")[-1].split("<|endofaction|>")[0].strip()

    return action_text


hotel_info = [
    "index",
    "地址",
    "區域",
    "網際網路",
    "停車處",
    "id",
    "location",
    "名稱",
    "電話",
    "郵編",
    "價格",
    "價格範圍",
    "星級",
    "takesbookings",
    "型別",
    "n",
    "預訂日期",
    "預定日期",
    "預定停留天數",
]
train_info = [
    "index",
    "到達時間",
    "日期",
    "出發地",
    "目的地",
    "時間",
    "出發時間",
    "價格",
    "列車號",
]
restaurant_info = [
    "index",
    "地址",
    "區域",
    "食物",
    "id",
    "introduction",
    "location",
    "名稱",
    "電話",
    "郵編",
    "價格範圍",
    "型別",
    "signature",
]
attraction_info = [
    "index",
    "地址",
    "區域",
    "費用",
    "id",
    "location",
    "名稱",
    "openhours",
    "電話",
    "郵編",
    "價格範圍",
    "型別",
]
database_keys = {
    "旅館": hotel_info,
    "列車": train_info,
    "餐廳": restaurant_info,
    "景點": attraction_info,
}

model_checkpoint = r"zhtw_rdrop_end2end_ver2_resid_03_alpha_30" # zhtw_rdrop_end2end_resid_03_alpha_30
tokenizer = BertTokenizerFast.from_pretrained(model_checkpoint)
model = GPT2LMHeadModel.from_pretrained(model_checkpoint, eos_token_id=102)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

domain_queue = []
context = ""
import random
seed = 123
torch.manual_seed(seed)
random.seed(seed)
def user_text(user_turn_context, history=[]):
    multiwoz_db = MultiWozDB()
    global context
    global domain_queue
    user = "<|user|> {}".format(user_turn_context)
    context = context + " " + user
    # turn_context = "可以幫我安排一個景點嗎?"#我想找到一家價格適中的餐廳
    turn_context = '[CLS] <|context|> {} <|endofcontext|>'.format(context)
    turn_context = turn_context.strip()
    input_text = [turn_context]
    tokens_tensor = tokenizer(
        input_text,
        add_special_tokens=False,
        truncation=True,
        padding=True,
        return_tensors="pt",
    )["input_ids"].to(device)

    beam_output = model.generate(
        tokens_tensor,
        num_beams=3,
        num_return_sequences=3,
        do_sample=True,
        max_length=1024,
        early_stopping=True,
        pad_token_id=102,
    )
    # print("beam_output", beam_output)
    # for i in range(3):
    #     print(i, "".join(tokenizer.decode(beam_output[i]).split()))
    tmp_pred = "".join(tokenizer.decode(beam_output[0], skip_special_tokens=False).split())
    # return tmp_pred
    # return turn_context
    pred_bs = get_belief(tmp_pred)
    pred_bs = clean_belief(pred_bs)
    beliefs, error_msg = convert_belief(pred_bs)
    action_text = get_action(tmp_pred)

    if beliefs:
        domain = get_turn_domain(beliefs, domain_queue)
    if pred_bs and not beliefs:
        # generated beliefs are incorrect
        # response_text = "對不起，我不明白你剛才說的話。請以有意義的方式重複。"
        lex_response = error_msg
    else:
        response_text = get_response_new(tmp_pred)

    db_results = multiwoz_db.queryResultVenues(domain, beliefs[domain], real_belief=True)

    if beliefs:
        results_dic = []
        for a in db_results:
            a_dic = dict.fromkeys(database_keys[domain])
            for k, v in zip(database_keys[domain], a):
                a_dic[k] = v
            results_dic.append(a_dic)
    if "!rest" in user_turn_context:
        beliefs, domain, domain_queue, history = [], [], [], []
    # %%
    if beliefs and domain == "餐廳":
        lex_response = lexicalize_restaurant(
            response_text, results_dic, beliefs, turn_domain=domain
        )

    if beliefs and domain == "景點":
        lex_response = lexicalize_attraction(
            response_text, results_dic, beliefs, turn_domain=domain
        )

    if beliefs and domain == "列車":
        lex_response = lexicalize_train(
            response_text, results_dic, beliefs, turn_domain=domain
        )

    if beliefs and domain == "旅館":
        lex_response = lexicalize_hotel(
            response_text, results_dic, beliefs, turn_domain=domain
        )
    system = '<|system|> {}'.format(lex_response)
    context = context + "" + system

    history.append((user_turn_context, lex_response))
    print("history", history)
    # print("beliefs:", beliefs)
    # print("tmp_pred", tmp_pred)
    # print("aaa", (tmp_pred, lex_response))
    # print(type((tmp_pred, lex_response)))
    html = "<div class='chatbot'>"
    for user_msg, resp_msg in history:
        html += f"<div class='user_msg'>{user_msg}</div>"
        html += f"<div class='resp_msg'>{resp_msg}</div>"
    html += "</div>"
    print(html)
    return history, history, beliefs, action_text, response_text
    # return html, history
    # return lex_response

if __name__ == "__main__":
    # text = input("Enter your text:")
    # aaa = user_text(text)
    # print("aaa:", aaa)

    # gr.Interface(fn=user_text,
    #              inputs=["text", "state"],
    #              outputs=["chatbot", "state"],
    #              allow_screenshot=False,
    #              allow_flagging="never",
    #              ).launch()

    # css = """
    # .chatbot {display:flex;flex-direction:column;width:90%;height:90%}
    # .msg {padding:4px;margin-bottom:4px;border-radius:4px;width:80%}
    # .user_msg {background-color:cornflowerblue;color:white;font-size:20px}
    # .resp_msg {background-color:lightgray;align-self:self-end;font-size:20px}
    # .footer {display:none !important}
    # """
    css = """.gradio-interface[theme=default] .panel-header {font-size:20px; text-align:center;}
             .h-64 {height:25rem;}
             .text-white {font-size:18px}
             .space-y-4>:not([hidden])~:not([hidden]) {font-size:18px}
    """

    title = "中文多領域任務導向對話系統"
    description = """<center>
    <img src="/images/lab_logo.png" width=350px>
    結合SimpleTOD和R-Drop演算法架構實現
    </center>"""
    gr.Interface(fn=user_text,
                 inputs=[gr.inputs.Textbox(placeholder="輸入你今天想要做什麼?", label="你的輸入"), "state"],
                 outputs=["state", gr.outputs.Chatbot(label="對話歷史"),
                          gr.outputs.Textbox(label="Belief State"),
                          gr.outputs.Textbox(label="System Action"),
                          gr.outputs.Textbox(label="delexicalized response")],
                 # outputs=["html", "state"],
                 allow_screenshot=False,
                 allow_flagging="never",
                 title=title,
                 description=description,
                 css=css
                 ).launch()