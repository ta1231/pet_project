import requests
import pandas as pd

csv_file_path = "/Users/teajun/Desktop/산종설/workspace/0427_두부/230427_data.CSV"
csv_url = "http://localhost:8000/train"

with open(csv_file_path, "rb") as f:
    file_content = f.read()
    # print(file_content)
    response = requests.post(csv_url, data={"file": file_content})

if response.status_code == 200:
    df = pd.DataFrame(response.json())
    print(df.head())
else:
    print("Error: ", response.status_code, response.content)
