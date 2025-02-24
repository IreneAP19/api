import requirements

url = "http://127.0.0.1:8000/predict"
data = {"tv": 100.0, "radio": 50.0, "newspaper": 25.0}

response = requirements.post(url, json=data)
print(response.json())  # Muestra la respuesta del servidor
