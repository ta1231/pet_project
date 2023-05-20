import requests

url = "http://127.0.0.1:8000/uploadfile/"

with open("/Users/teajun/Desktop/산종설/workspace/0427_두부/230427_data.CSV", "rb") as f:
    files = {"file": f}
    response = requests.post(url, files=files)

print(response.json())
