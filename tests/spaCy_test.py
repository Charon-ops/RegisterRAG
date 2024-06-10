import urllib.parse
import spacy
import random
import hashlib
import urllib
from http import client
import json
import time


def translate(query: str):
    baidu_app_id = 20230110001525993
    baidu_api_key = "KZY_UYhS_I7sQ5THfvIy"

    sault = random.randint(32768, 65536)
    sign = f"{baidu_app_id}{query}{sault}{baidu_api_key}"
    sign_md5 = hashlib.md5(sign.encode()).hexdigest()

    url = f"/api/trans/vip/translate?appid={baidu_app_id}&q={urllib.parse.quote(query)}&from=auto&to=en&salt={sault}&sign={sign_md5}"

    try:
        httpClient = client.HTTPConnection("fanyi-api.baidu.com")
        httpClient.request("GET", url)
        response = httpClient.getresponse()
        if response.status == 200:
            result = response.read().decode("utf-8")
            res_json = json.loads(result)
            return res_json["trans_result"][0]["dst"]
    except Exception as e:
        print("Error:" + e)


start_time = time.time()
query_english = translate("如何在工业发酵过程中优化微生物的生长和代谢产物的产量？")
print(query_english)
nlp = spacy.load("en_core_web_sm", exclude=["ner"])
doc = nlp(query_english)
keywords = [
    token.text
    for token in doc
    if not token.is_stop and not token.is_punct and token.pos_ in {"NOUN", "VERB"}
]
keywords_clean = [keyword for keyword in keywords if keyword.isalnum()]
end_time = time.time()

print(f"Time: {end_time - start_time}")
print(keywords_clean)
