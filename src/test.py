import requests

url = 'http://localhost:8080/2015-03-31/functions/function/invocations'
data = {'url': 'https://raw.githubusercontent.com/SartajBhuvaji/Brain-Tumor-Classification-DataSet/master/Testing/no_tumor/image(103).jpg'}

result = requests.post(url=url, json=data).json()
print(result)
