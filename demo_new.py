# %%
from transformers import BertTokenizerFast, GPT2LMHeadModel
import torch

from evaluate_multiwoz_zhtw import MultiWozDB
from utils.simpletod import get_belief, clean_belief, convert_belief, get_response_new
from lexicalize_response import lexicalize_restaurant, lexicalize_attraction, lexicalize_hotel, lexicalize_train

# %%


def get_turn_domain(beliefs, q):
    for k in beliefs.keys():
        if k not in q:
            q.append(k)
            return k
    return q[-1]


# %%
# model_checkpoint = r"test6endtoend\reinforce_0_ete_0.6"
# model_checkpoint = r"D:\pycharm_project\demo_for_nlp\simpleTOD-zhtw\zhtw_baseline_end2end"
model_checkpoint = r"D:\pycharm_project\demo_for_nlp\simpleTOD-zhtw\delex_end2end_output_with_none_repeat_action"
tokenizer = BertTokenizerFast.from_pretrained(model_checkpoint)
model = GPT2LMHeadModel.from_pretrained(model_checkpoint, eos_token_id=102)
model.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print("model loaded")
multiwoz_db = MultiWozDB()

# %%
# turn_context = "[CLS] <|context|> <|user|> 你好，我想在市中心找家餐廳，要價格便宜並且提供土耳其美食的。 <|system|>目前沒有符合這些條件的餐廳。<|user|>請推薦市中心的一家餐廳。<|system|>我有兩家符合該條件的餐廳，安納托裡亞酒店和艾菲斯餐廳。您要我為您預訂一間嗎？<|user|>請為我預定艾菲斯餐廳，謝謝。<|endofcontext|>"
# turn_context = "[CLS]<|context|><|user|>我想找到一家價格適中的餐廳。<|endofcontext|>"
turn_context = "我想找到一家價格適中的餐廳"#我想找到一家價格適中的餐廳可以幫我安排一個景點嗎?
turn_context = '[CLS] <|context|> {} <|endofcontext|>'.format(turn_context)
print("turn_context:", turn_context)
print("model_eos_token:", tokenizer.eos_token_id)
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
    num_beams=1,
    num_return_sequences=1,
    max_length=1024,
    early_stopping=True,
    pad_token_id=102,
)
print("beam_output:", beam_output)
tmp_pred = "".join(tokenizer.decode(beam_output[0], skip_special_tokens=False).split()) #把每個字之間的空白去除
print("tmp_pred", tmp_pred)
pred_bs = get_belief(tmp_pred)
pred_bs = clean_belief(pred_bs)
beliefs, error_msg = convert_belief(pred_bs)
domain_queue = []
domain = ""
if beliefs:
    domain = get_turn_domain(beliefs, domain_queue)


if pred_bs and not beliefs:
    # generated beliefs are incorrect
    # response_text = "對不起，我不明白你剛才說的話。請以有意義的方式重複。"
    lex_response = error_msg
else:
    response_text = get_response_new(tmp_pred)


# %%
if beliefs:
    db_results = multiwoz_db.queryResultVenues(
        domain, beliefs[domain], real_belief=True
    )
print("db_results:", db_results)
# %%
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

# %%
if beliefs:
    results_dic = []
    for a in db_results:
        a_dic = dict.fromkeys(database_keys[domain])
        for k, v in zip(database_keys[domain], a):
            a_dic[k] = v
        results_dic.append(a_dic)


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

print("lex_response:", lex_response)
# %%
