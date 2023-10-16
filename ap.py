from flask import Flask, request, render_template, redirect, url_for
from facenet_pytorch import InceptionResnetV1
import cv2
import numpy as np
from mtcnn.mtcnn import MTCNN
import torch
import pymysql
from io import BytesIO
import base64

app = Flask(__name__)

# Cargar el modelo InceptionResnetV1
encoder = InceptionResnetV1(pretrained='vggface2', classify=False).eval()

# Conectar a la base de datos MySQL
conn = pymysql.connect(host='localhost', user='root', password='', database='app-recognition')
cursor = conn.cursor()

# Ruta para el formulario
@app.route('/index')
def index():
    return render_template('my-app/templates/public/appform/form.html')

# Ruta para procesar el formulario
@app.route('/procesar', methods=['POST'])
def procesar():
    nombre_usuario = request.form['nombre_usuario']
    imagen = request.files['imagen']

    # Procesar la imagen y almacenar el descriptor facial
    descriptor_bytes = procesar_imagen(imagen)

    if descriptor_bytes is not None:
        # Almacenar el descriptor facial en la base de datos junto con el nombre de usuario
        insert_query = "INSERT INTO tabla_descriptores (descriptor_data, nombre_usuario) VALUES (%s, %s)"
        cursor.execute(insert_query, (descriptor_bytes, nombre_usuario))
        conn.commit()  # Guardar los cambios en la base de datos
        return "Descriptor facial almacenado en la base de datos para el usuario " + nombre_usuario
    else:
        return "No se detectaron caras en la imagen o hubo un error en el procesamiento."

def procesar_imagen(imagen):
    # Cargar la imagen
    image = cv2.imdecode(np.fromstring(imagen.read(), np.uint8), cv2.IMREAD_COLOR)

    # Crear una instancia de MTCNN
    mtcnn = MTCNN()

    # Detectar caras en la imagen
    caras = mtcnn.detect_faces(image)

    if len(caras) > 0:
        # Extraer la región de la cara
        x, y, width, height = caras[0]['box']
        face = image[y:y+height, x:x+width]

        # Redimensionar la cara a 160x160
        face = cv2.resize(face, (160, 160))

        # Preprocesar la imagen para el modelo InceptionResnetV1
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = np.transpose(face, (2, 0, 1))
        face = face / 255.0
        face_tensor = torch.tensor(face, dtype=torch.float32)
        face_tensor = face_tensor.unsqueeze(0)

        # Generar el embedding de la cara
        embedding_cara = encoder(face_tensor)

        # Convertir el embedding de la cara en un arreglo NumPy
        embedding_array = embedding_cara.detach().numpy()

        # Convertir el arreglo NumPy en bytes
        descriptor_bytes = embedding_array.tobytes()
        return descriptor_bytes
    else:
        return None

if __name__ == '__main__':
    app.run(debug=True)
