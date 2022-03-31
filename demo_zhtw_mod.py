
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer, BertTokenizerFast
import sys, os
import json
from collections import Counter
import sqlite3
import ipdb
import random
from utils.multiwoz.nlp import normalize, normalize_for_sql
from evaluate_multiwoz_zhtw import MultiWozDB
from lexicalize_response import lexicalize_restaurant, lexicalize_attraction, lexicalize_hotel, lexicalize_train
from utils.simpletod import get_belief, clean_belief, convert_belief, get_response_new
import pprint

import logging
import time

from colorama import Fore, Back, Style

logging.basicConfig(level=logging.INFO)
logging.getLogger("transformers.file_utils").setLevel(logging.ERROR)
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
logging.getLogger("transformers.modeling_gpt2").setLevel(logging.ERROR)
logging.getLogger("transformers.configuration_utils").setLevel(logging.ERROR)
logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)


def get_belief_new_dbsearch(sent):
    if "<|belief|>" in sent:
        tmp = sent.strip(" ").split("<|belief|>")[-1].split("<|endofbelief|>")[0]
    else:
        return []
    # else:
    #     raise TypeError('unknown belief separator')
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
    # if '<|response|>' in sent:
    #     tmp = sent.split('<|belief|>')[-1].split('<|action|>')[-1].split('<|response|>')[-1]
    # else:
    #     return ''
    # tmp = tmp.strip(' .,')
    # # assert tmp.endswith('<endofresponse>')
    # tmp = tmp.replace('<|endofresponse|>', '')
    # tmp = tmp.replace('[CLS]', '')
    # tokens = tokenizer.encode(tmp)
    # new_tokens = []
    # for tok in tokens:
    #     # if tok in break_tokens:
    #     if tok in tokenizer.encode("[SEP]"):#if tok in tokenizer.encode(tokenizer._eos_token):
    #         continue
    #     new_tokens.append(tok)
    # # ipdb.set_trace()
    # response = tokenizer.decode(new_tokens).strip(' ,.')
    # # response = "".join(response.split()) #把每個字的空格去掉
    # return response


def get_turn_domain(beliefs, q):
    for k in beliefs.keys():
        if k not in q:
            q.append(k)
            return k
    return q[-1]

pp = pprint.PrettyPrinter(indent=4)
prev_beliefs = {}
domain_queue = []
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
if __name__ == '__main__':

    print('\33]0;SimpleTOD\a', end='')
    sys.stdout.flush()

    # model_checkpoint = sys.argv[1]
    # model_checkpoint = r"delex_end2end_output_with_none_repeat_action"
    model_checkpoint = r"zhtw_rdrop_end2end" #zhtw_rdrop_end2end_resid_03_alpha_25
    # decoding = sys.argv[2]
    decoding = 'nucleus'
    TOP_P = float(2)
    # if decoding == 'nucleus':
    #     TOP_P = float(sys.argv[3])

    delay = 0.5
    multiwoz_db = MultiWozDB()

    print('\nLoading Model', end="")
    print("\nThe model_checkpoint is {}".format(model_checkpoint), end="")

    # tokenizer = GPT2Tokenizer.from_pretrained(model_checkpoint)
    tokenizer = BertTokenizerFast.from_pretrained(model_checkpoint)
    model = GPT2LMHeadModel.from_pretrained(model_checkpoint)

    # model.load_state_dict(torch.load(model_checkpoint))
    model.eval()
    model.to('cuda')

    # break_tokens = tokenizer.encode(tokenizer._eos_token) + tokenizer.encode('?') + tokenizer.encode('!')
    break_tokens = tokenizer.encode('?') + tokenizer.encode('!')
    # break_tokens = tokenizer.encode(tokenizer._eos_token)
    MAX_LEN = model.config.n_ctx

    sample = 1
    print()
    print(Fore.MAGENTA + '\n中文SimpleTOD已準備好了。你想問什麼?' + Style.RESET_ALL)
    # history = []
    context = ''
    input_text = ''
    turn = 0
    # dbmatch = 0

    while True:
        print(Fore.GREEN)
        raw_text = input('使用者: ')
        print(Style.RESET_ALL)
        input_text = raw_text.replace('you> ', '')
        if input_text in ['q', 'quit']:
            break
        user = '<|user|> {}'.format(input_text)
        context = context + ' ' + user
        text = '[CLS] <|context|> {} <|endofcontext|>'.format(context)

        # print(context)

        text = text.strip()
        indexed_tokens = tokenizer.encode(text, add_special_tokens=False)

        if len(indexed_tokens) > MAX_LEN:
            indexed_tokens = indexed_tokens[-1*MAX_LEN:]
        # Convert indexed tokens in a PyTorch tensor
        tokens_tensor = torch.tensor([indexed_tokens])

        # If you have a GPU, put everything on cuda
        tokens_tensor = tokens_tensor.to('cuda')
        predicted_index = indexed_tokens[-1]

        # if decoding == 'nucleus':
        #     sample_output = model.generate(
        #         tokens_tensor,
        #         do_sample=True,
        #         max_length=MAX_LEN,
        #         top_p=TOP_P,
        #         top_k=0
        #     )
        # elif decoding == 'greedy':
        #     sample_output = model.generate(
        #         tokens_tensor,
        #         max_length=MAX_LEN,
        #         do_sample=False
        #     )
        # predicted_text = tokenizer.decode(sample_output[0], skip_special_tokens=True)


        with torch.no_grad():
            while predicted_index not in break_tokens:
                outputs = model(tokens_tensor)
                predictions = outputs[0]
                predicted_index = torch.argmax(predictions[0, -1, :]).item()
                indexed_tokens += [predicted_index]
                tokens_tensor = torch.tensor([indexed_tokens]).to('cuda')
                if len(indexed_tokens) > MAX_LEN:
                    break
                if tokenizer.decode(indexed_tokens).endswith('<|endofbelief|>'):
                    break

        tmp_pred = tokenizer.decode(indexed_tokens)  #我想找到一家價格適中的餐廳
        ## print("tmp_pred:", tmp_pred)
        belief_text = get_belief_new_dbsearch(tmp_pred)
        # print("0_belief_text:", belief_text)
        # print(tmp_pred)
        belief_text = clean_belief(belief_text)
        # print("1_belief_text:", belief_text)
        beliefs, error_msg = convert_belief(belief_text)
        # print("beliefsss:", beliefs)
        # print("error_msg:", error_msg)
        # print("beliefs:", beliefs)
        # domain = list(beliefs.keys())[0]
        domain = get_turn_domain(beliefs, domain_queue)



        # continue generation after creating db
        indexed_tokens = tokenizer.encode(text, add_special_tokens=False) #新增add_special_tokens=False
        ## print("indexed_tokens:", indexed_tokens)
        ## print("after_decode_index_:", tokenizer.decode(indexed_tokens))

        if len(indexed_tokens) > MAX_LEN:
            indexed_tokens = indexed_tokens[-1 * MAX_LEN:]

        # Convert indexed tokens in a PyTorch tensor
        tokens_tensor = torch.tensor([indexed_tokens])

        # If you have a GPU, put everything on cuda
        tokens_tensor = tokens_tensor.to('cuda')
        predicted_index = indexed_tokens[-1]

        truncate_action = False
        # Predict all tokens
        with torch.no_grad():
            while predicted_index not in break_tokens:
                outputs = model(tokens_tensor)
                # print("model_output:", type(outputs))
                predictions = outputs[0]
                predicted_index = torch.argmax(predictions[0, -1, :]).item()
                indexed_tokens += [predicted_index]
                if len(indexed_tokens) > MAX_LEN:
                    break


                predicted_text = tokenizer.decode(indexed_tokens)
                # print("First_predicted_text:", predicted_text)
                if '<|action|>' in predicted_text:
                    generated_actions = predicted_text.split('<|action|>')[-1].split('<|endofaction|>')[0].split(',')
                    new_actions = []
                    for a in generated_actions:
                        if a in ['', ' ']:
                            continue
                        new_actions.append(a.strip())
                    len_actions = len(new_actions)
                    if len(list(set(new_actions))) > len(new_actions) or (len_actions > 10 and not truncate_action):
                        # ipdb.set_trace()
                        actions = '<|action|> {} <|endofaction|>'.format(' , '.join(list(set(new_actions))))
                        indexed_tokens = tokenizer.encode('{} {}'.format(predicted_text.split('<|action|>')[0], actions))
                        # print('action truncated')
                        truncate_action = True
                tokens_tensor = torch.tensor([indexed_tokens]).to('cuda')

        predicted_text = "".join(tokenizer.decode(indexed_tokens).split())
        # print("Second_predicted_text:", predicted_text)
        # predicted_text = "".join(predicted_text.split())  #把每個字的空格去掉
        # print("remove_space_predicted_text:", predicted_text)
        # action_text = get_action_new(predicted_text)
        response_text = get_response_new(predicted_text)
        ## print("response_text:", response_text)
        ## print(predicted_text)

        db_results = multiwoz_db.queryResultVenues(domain, beliefs[domain], real_belief=True)

###新增部分

        if beliefs:
            results_dic = []
            for a in db_results:
                a_dic = dict.fromkeys(database_keys[domain])
                for k, v in zip(database_keys[domain], a):
                    a_dic[k] = v
                results_dic.append(a_dic)

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

        ## print("lex_response:", lex_response)
        # if domain == 'train':
        #     lex_response = lexicalize_train(response_text, db_results, beliefs, turn_domain=domain)
        # elif domain == 'hotel':
        #     lex_response = lexicalize_hotel(response_text, db_results, beliefs, turn_domain=domain)
        # else:
        #     ipdb.set_trace()
        #     raise TypeError('unknown domain')

        delex_system = '<|system|> {}'.format(response_text)
        system = '<|system|> {}'.format(lex_response)
        context = context + ' ' + system


        print(Fore.CYAN + 'SimpleTOD: ', end="")
        for a in lex_response.split(' '):
            print(a + ' ', end="")
            sys.stdout.flush()
            time.sleep(delay)
        print(Style.RESET_ALL)
        print(Fore.YELLOW + 'belief: {}'.format(beliefs) + Style.RESET_ALL)

        print(Style.RESET_ALL)

        turn += 1
        prev_beliefs = beliefs

