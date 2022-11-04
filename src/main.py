from src.spaceship_titanic import predecir, DatosEntrada
from pathlib import Path

datos_entrada: DatosEntrada = {
    'CryoSleep': [False],
    'FoodCourt': [9],
    'VIP': [False],
    'Age': [19],
    'Spa': [2823],
    'VRDeck': [0],
    'Destination': ['TRAPPIST-1e'],
    'HomePlanet': ['Earth'],
    'RoomService': [0],
    'ShoppingMall': [0],
    'Transported': [True]
}
project_path = Path(__file__).parent.parent.resolve().absolute()
prediccion = predecir(datos_entrada=datos_entrada, project_path=project_path)
print(prediccion)
