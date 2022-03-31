import random


def lexicalize_restaurant(delex_response, results_dic, turn_beliefs, turn_domain):
    if len(results_dic) > 0:
        sample = random.sample(results_dic, k=1)[0]
        value_count = len(results_dic)
    else:
        sample = turn_beliefs[turn_domain]
        value_count = 0

    # print(sample)
    lex_response = delex_response

    if "[value_count]家" in delex_response and not results_dic:
        lex_response = "目前沒有符合這些條件的餐廳。"
        return lex_response

    if (
        "[restaurant_name]是[value_food]，[restaurant_name]是[value_food]"
        in delex_response
    ):
        lex_response = lex_response.replace(
            "[restaurant_name]是[value_food]，[restaurant_name]是[value_food]",
            "[restaurant_name]是[value_food]餐廳。",
        )
    if "[restaurant_address]" in delex_response:
        lex_response = lex_response.replace("[restaurant_address]", sample["地址"])
    if "[restaurant_name]" in delex_response:
        lex_response = lex_response.replace("[restaurant_name]", sample["名稱"])
    if "[restaurant_phone]" in delex_response:
        phone = str(int(sample["電話"]))
        lex_response = lex_response.replace("[restaurant_phone]", phone)
    if "[restaurant_postcode]" in delex_response:
        lex_response = lex_response.replace("[restaurant_postcode]", sample["郵編"])
    if "[restaurant_reference]" in delex_response:
        random_number = random.randint(10000, 99999)
        lex_response = lex_response.replace(
            "[restaurant_reference]", str(random_number)
        )
    if "[value_area]" in delex_response:
        lex_response = lex_response.replace("[value_area]", sample["區域"])
    if "[value_pricerange]" in delex_response:
        lex_response = lex_response.replace("[value_pricerange]", sample["價格範圍"])
    if "[value_count]分鐘" in delex_response:
        lex_response = lex_response.replace("[value_count]分鐘", "15分鐘")
    if "[value_count]人" in delex_response:
        people = sample["預定人數"]
        lex_response = lex_response.replace("[value_count]人", people)
    if "[value_count]" in delex_response:
        lex_response = lex_response.replace("[value_count]", str(value_count))
    if "另[value_count]個" in delex_response:
        lex_response = lex_response.replace("另[value_count]個", "另一個")
    if "有[value_count]個" in delex_response:
        lex_response = lex_response.replace("有[value_count]個", f"有{str(value_count)}個")

    if "[value_day]" in delex_response:
        day = sample["預定日期"]
        lex_response = lex_response.replace("[value_day]", day)
    if "[value_food]" in delex_response:
        day = sample["食物"]
        lex_response = lex_response.replace("[value_food]", day)
    if "[value_food]義大利餐廳" in delex_response:
        lex_response = lex_response.replace("[value_food]義大利餐廳", "義大利餐廳")
    if "您喜歡[value_food]還是[value_food]" in delex_response:
        lex_response = lex_response.replace(
            "您喜歡[value_food]還是[value_food]？", "您想要其中一間嗎？"
        )

    # Duplicate error
    if "您想幾點用餐，您想幾點用餐？" in lex_response:
        lex_response = lex_response.replace("您想幾點用餐，您想幾點用餐？", "您想幾點用餐？")

    return lex_response


def lexicalize_attraction(delex_response, results_dic, turn_beliefs, turn_domain):
    if len(results_dic) > 0:
        sample = random.sample(results_dic, k=1)[0]
        value_count = len(results_dic)
    else:
        sample = turn_beliefs[turn_domain]
        value_count = 0

    lex_response = delex_response

    if "[value_count]家" in delex_response and not results_dic:
        lex_response = "目前沒有符合這些條件的餐廳。"
        return lex_response

    if "[value_area]有[value_count]家劇院，[value_area]有[value_count]家劇院" in delex_response:
        lex_response = lex_response.replace(
            "[value_area]有[value_count]家劇院，[value_area]有[value_count]家劇院",
            "[value_area]有[value_count]家劇院",
        )

    if "[attraction_address]" in delex_response:
        lex_response = lex_response.replace("[attraction_address]", sample["地址"])
    if "[attraction_name]" in delex_response:
        lex_response = lex_response.replace("[attraction_name]", sample["名稱"])
    if "[attraction_phone]" in delex_response:
        phone = str(int(sample["電話"]))
        lex_response = lex_response.replace("[attraction_phone]", phone)
    if "[attraction_postcode]" in delex_response:
        lex_response = lex_response.replace("[attraction_postcode]", sample["郵編"])
    if "[value_area]" in delex_response:
        lex_response = lex_response.replace("[value_area]", sample["區域"])

    if "[value_count]個[value_count]個[value_count]個[value_count]個的" in delex_response:
        lex_response = lex_response.replace(
            "[value_count]個[value_count]個[value_count]個[value_count]個的", "其中一個"
        )
    if "[value_count]個[value_count]個[value_count]個" in delex_response:
        lex_response = lex_response.replace(
            "[value_count]個[value_count]個[value_count]個", "[value_count]個"
        )
        lex_response = lex_response.replace("[value_count]個", str(value_count))
    if "[value_count]個[value_count]個" in delex_response:
        lex_response = lex_response.replace("[value_count]個[value_count]個", "")
    if "另[value_count]個" in delex_response:
        lex_response = lex_response.replace("另[value_count]個", "另一個")
    if "有[value_count]個" in delex_response:
        lex_response = lex_response.replace("有[value_count]個", f"有{str(value_count)}個")
    if "[value_pricerange]" in delex_response:
        lex_response = lex_response.replace("[value_pricerange]", sample["價格範圍"])

    return lex_response


def lexicalize_hotel(delex_response, results_dic, turn_beliefs, turn_domain):
    if len(results_dic) > 0:
        sample = random.sample(results_dic, k=1)[0]
        value_count = len(results_dic)
    else:
        sample = turn_beliefs[turn_domain]
        value_count = 0

    # print(sample)
    lex_response = delex_response

    if "[value_count]家" in delex_response and not results_dic:
        lex_response = "目前沒有符合這些條件的餐廳。"
        return lex_response

    if "[hotel_address]" in delex_response:
        lex_response = lex_response.replace("[hotel_address]", sample["地址"])
    if "[hotel_name]" in delex_response:
        lex_response = lex_response.replace("[hotel_name]", sample["名稱"])
    if "[hotel_phone]" in delex_response:
        phone = str(int(sample["電話"]))
        lex_response = lex_response.replace("[hotel_phone]", phone)
    if "[hotel_postcode]" in delex_response:
        lex_response = lex_response.replace("[hotel_postcode]", sample["郵編"])
    if "[value_area]" in delex_response:
        lex_response = lex_response.replace("[value_area]", sample["區域"])
    if "[value_count]晚" in delex_response:
        day = sample["預定停留天數"]
        lex_response = lex_response.replace("[value_count]晚", day)
    if "[value_count]人" in delex_response:
        people = sample["預定人數"]
        lex_response = lex_response.replace("[value_count]人", people)
    if "[value_count]" in delex_response:
        lex_response = lex_response.replace("[value_count]", str(value_count))
    if "[hotel_reference]" in delex_response:
        random_number = random.randint(10000, 99999)
        lex_response = lex_response.replace("[hotel_reference]", str(random_number))
    if "[value_pricerange]" in delex_response:
        lex_response = lex_response.replace("[value_pricerange]", sample["價格範圍"])
    if "[value_count]個[value_count]個" in delex_response:
        lex_response = lex_response.replace("[value_count]個[value_count]個", "")
    if "您想要[value_count]星" in delex_response:
        lex_response = lex_response.replace("您想要[value_count]星", "您想要幾星")
    if "[value_count]個[value_count]星和[value_count]個[value_count]星" in delex_response:
        lex_response = lex_response.replace(
            "[value_count]個[value_count]星和[value_count]個[value_count]星", "一個4星和一個3星"
        )
    if "[value_count]星" in delex_response:
        lex_response = lex_response.replace("[value_count]星", f"{sample['星級']}星")
    if "另[value_count]個" in delex_response:
        lex_response = lex_response.replace("另[value_count]個", "另一個")
    if "有[value_count]個" in delex_response:
        lex_response = lex_response.replace("有[value_count]個", f"有{str(value_count)}個")
    if "[value_day]" in delex_response:
        day = sample["預定日期"]
        lex_response = lex_response.replace("[value_day]", day)

    return lex_response


def lexicalize_train(delex_response, results_dic, turn_beliefs, turn_domain):
    if len(results_dic) > 0:
        sample = random.sample(results_dic, k=1)[0]
        value_count = len(results_dic)
    else:
        sample = turn_beliefs[turn_domain]
        value_count = 0

    # print(sample)
    lex_response = delex_response

    if "[train_id]" in delex_response:
        lex_response = lex_response.replace("[train_id]", sample["列車號"])
    if "[train_reference]" in delex_response:
        random_number = random.randint(10000, 99999)
        lex_response = lex_response.replace("[train_reference]", str(random_number))

    if "於[value_time]離開[value_place]" in delex_response:
        lex_response = lex_response.replace(
            "於[value_time]離開[value_place]", f"於{sample['出發時間']}離開[value_place]"
        )
        lex_response = lex_response.replace("離開[value_place]", f"離開{sample['出發地']}")
    if "於[value_time]到達[value_place]" in delex_response:
        lex_response = lex_response.replace(
            "於[value_time]到達[value_place]", f"於{sample['到達時間']}到達[value_place]"
        )
        lex_response = lex_response.replace("到達[value_place]", f"到達{sample['目的地']}")
    if "在[value_time]離開[value_place]" in delex_response:
        lex_response = lex_response.replace(
            "在[value_time]離開[value_place]", f"在{sample['出發時間']}離開[value_place]"
        )
        lex_response = lex_response.replace("離開[value_place]", f"離開{sample['出發地']}")
    if "在[value_time]到達[value_place]" in delex_response:
        lex_response = lex_response.replace(
            "在[value_time]到達[value_place]", f"在{sample['到達時間']}到達[value_place]"
        )
        lex_response = lex_response.replace("到達[value_place]", f"到達{sample['目的地']}")
    if "[value_time]離開[value_place]" in delex_response:
        lex_response = lex_response.replace(
            "[value_time]離開[value_place]", f"{sample['出發時間']}離開[value_place]"
        )
        lex_response = lex_response.replace("離開[value_place]", f"離開{sample['出發地']}")
    if "[value_time]到達[value_place]" in delex_response:
        lex_response = lex_response.replace(
            "[value_time]到達[value_place]", f"{sample['到達時間']}到達[value_place]"
        )
        lex_response = lex_response.replace("到達[value_place]", f"到達{sample['目的地']}")
    if "從[value_place]出發" in delex_response:
        lex_response = lex_response.replace("從[value_place]出發", f"從{sample['出發地']}出發")
    if "前往[value_place]" in delex_response:
        lex_response = lex_response.replace("前往[value_place]", f"前往{sample['目的地']}")
    if "在[value_time]出發" in delex_response:
        lex_response = lex_response.replace("在[value_time]出發", f"在{sample['出發時間']}出發")
    if "在[value_time]到達" in delex_response:
        lex_response = lex_response.replace("在[value_time]到達", f"在{sample['到達時間']}到達")

    if "[value_day]" in delex_response:
        train_day = sample["日期"]
        lex_response = lex_response.replace("[value_day]", train_day)
    if "[value_price]" in delex_response:
        train_price = sample["價格"]
        lex_response = lex_response.replace("[value_price]", train_price)

    return lex_response
