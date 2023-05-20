# import requests

# url = "http://localhost:8000/predict"
# files = {"input_file": ("file.pickle", open("../input.pkl", "rb"))}
# response = requests.post(url, files=files)


# import requests

# url = "http://localhost:8000/predict"

# with open("../input.pkl", "rb") as f:
#     input_file = f.read()

# files = {"input_file": input_file}
# print(files)
# response = requests.post(url, files=files)

# print(response.json())


import requests
import numpy as np
import json
import pickle

with open("../input.pkl", "rb") as f:
    var = pickle.load(f)

input_json = json.dumps(var.tolist())
response = requests.post("http://localhost:8000/predict_str", json=input_json)
print(response.json())