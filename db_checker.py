#%%
import json

domains_map = {"餐廳": "restaurant", "旅館": "hotel", "景點": "attraction", "列車": "train"}
domains = ["餐廳", "旅館", "景點", "列車"]

db_ckecker = {domain: {} for domain in domains}

# %%
with open(
    r"utils/multiwoz/db/zhtw_db/zhtw_attraction_db.json",
    encoding="UTF-8",
) as f:
    attractions = json.load(f)

slots = ["名稱"]
for slot in slots:
    if slot not in db_ckecker["景點"]:
        db_ckecker["景點"][slot] = []

for attraction in attractions:
    for slot in slots:
        db_ckecker["景點"][slot].append("".join(attraction[slot].strip().lower().split()))
        db_ckecker["景點"][slot] = list(set(db_ckecker["景點"][slot]))


# %%
with open(
    r"utils/multiwoz/db/zhtw_db/zhtw_restaurant_db.json",
    encoding="UTF-8",
) as f:
    restaurants = json.load(f)

slots = ["名稱"]
for slot in slots:
    if slot not in db_ckecker["餐廳"]:
        db_ckecker["餐廳"][slot] = []

for restaurant in restaurants:
    for slot in slots:
        db_ckecker["餐廳"][slot].append("".join(restaurant[slot].strip().lower().split()))
        db_ckecker["餐廳"][slot] = list(set(db_ckecker["餐廳"][slot]))

#%%
with open(
    r"utils/multiwoz/db/zhtw_db/zhtw_hotel_db.json",
    encoding="UTF-8",
) as f:
    hotels = json.load(f)

slots = ["名稱"]
for slot in slots:
    if slot not in db_ckecker["旅館"]:
        db_ckecker["旅館"][slot] = []

for hotel in hotels:
    for slot in slots:
        db_ckecker["旅館"][slot].append("".join(hotel[slot].strip().lower().split()))
        db_ckecker["旅館"][slot] = list(set(db_ckecker["旅館"][slot]))

#%%
with open(
    r"utils/multiwoz/db/zhtw_db/zhtw_train_db.json",
    encoding="UTF-8",
) as f:
    trains = json.load(f)

slots = ["出發地", "目的地"]
for slot in slots:
    if slot not in db_ckecker["列車"]:
        db_ckecker["列車"][slot] = []

for train in trains:
    for slot in slots:
        db_ckecker["列車"][slot].append("".join(train[slot].strip().lower().split()))
        db_ckecker["列車"][slot] = list(set(db_ckecker["列車"][slot]))

#%%
# save file

with open("db_checker.json", "wt", encoding="UTF-8") as f:
    json.dump(db_ckecker, f, ensure_ascii=False, indent=4)
# %%
