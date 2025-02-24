import requests

import requests

def test_ingest_endpoint():
    url = 'http://localhost:8000/ingest'  
    data = {'data': [[100, 100, 200, 3000], [200, 230, 500, 4000]]}
    response = requests.post(url, json=data)
    
    # Imprimir el código de estado y la respuesta
    print(f"Status Code: {response.status_code}")
    print(f"Response Body: {response.text}")
    
    # Verificar la respuesta esperada
    assert response.status_code == 200
    assert response.json() == {'message': 'Datos ingresados correctamente'}

if __name__ == '__main__':
    test_ingest_endpoint()


def test_predict_endpoint():
    url = 'http://localhost:8000/predict'
    data = {'data': [[100, 100, 200]]}  # Datos a enviar
    
    response = requests.post(url, json=data)
    assert response.status_code == 200
    assert 'prediction' in response.json()

if __name__ == '__main__':
    test_predict_endpoint()

def test_retrain_endpoint():
    url = 'http://localhost:8000/retrain'
    response = requests.post(url)
    assert response.status_code == 200
    assert response.json() == {'message': 'Modelo reentrenado correctamente.'}
    
if __name__ == '__main__':
    test_retrain_endpoint()
