from fastapi import FastAPI, HTTPException
import numpy as np
import joblib





# Cargar el modelo previamente entrenado
def load_model():
    model_path = "model/model.pkl"  # Ruta donde se almacena el modelo entrenado
    model = joblib.load(model_path)
    return model

# Función para predecir las ventas a partir de los gastos en marketing
def predict_sales(tv, radio, newspaper):
    model = load_model()  # Cargar el modelo
    features = np.array([[tv, radio, newspaper]])  # Los valores de entrada deben estar en un array 2D
    prediction = model.predict(features)[0]  # Realizar la predicción
    return prediction
    
# 2. Endpoint de ingesta de datos
from sqlalchemy import create_engine, Column, Float, Integer
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

DATABASE_URL = "sqlite:///./sales_data.db"  # Usamos SQLite para simplificar

# Crear la base de datos y la tabla
Base = declarative_base()
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Definición de la tabla de ventas
class SalesData(Base):
    __tablename__ = "sales_data"
    id = Column(Integer, primary_key=True, index=True)
    tv = Column(Float)
    radio = Column(Float)
    newspaper = Column(Float)
    sales = Column(Float, nullable=True)  # De momento sin ventas, ya que se predicen

Base.metadata.create_all(bind=engine)

# Función para guardar datos en la base de datos
def ingest_data(tv, radio, newspaper):
    db = SessionLocal()  # Crear una sesión para interactuar con la base de datos
    
    # Crear un nuevo registro con los datos recibidos
    new_record = SalesData(tv=tv, radio=radio, newspaper=newspaper)
    db.add(new_record)
    db.commit()  # Guardar los cambios
    db.refresh(new_record)
    db.close()  # Cerrar la sesión
    
    return new_record

# 3. Endpoint de reentramiento del modelo


import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression

# Función para reentrenar el modelo con los datos de la base de datos
def retrain_model():
    db = SessionLocal()  # Crear una sesión para interactuar con la base de datos
    
    # Obtener todos los registros de la base de datos
    records = db.query(SalesData).all()
    db.close()  # Cerrar la sesión

    # Convertir los registros en un DataFrame
    df = pd.DataFrame(records, columns=["tv", "radio", "newspaper", "sales"])

    # Asegurarse de que hay suficientes datos
    if len(df) < 2:  # Si hay pocos registros, no podemos entrenar el modelo
        raise ValueError("Not enough data to retrain the model.")
    
    # Definir las características y la variable objetivo
    X = df[["tv", "radio", "newspaper"]]
    y = df["sales"]

    # Entrenar el modelo de regresión lineal
    model = LinearRegression()
    model.fit(X, y)

    # Guardar el modelo entrenado
    joblib.dump(model, "model/model.pkl")
    
    return {"message": "Model retrained successfully"}
