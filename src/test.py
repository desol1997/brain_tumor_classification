import requests

url = 'https://sh1oyyzmr4.execute-api.us-east-1.amazonaws.com/test/predict'
data = {'url': 'https://raw.githubusercontent.com/SartajBhuvaji/Brain-Tumor-Classification-DataSet/master/Testing/no_tumor/image(103).jpg'}

result = requests.post(url=url, json=data).json()
print(result)
