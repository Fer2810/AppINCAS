from flask import Flask, request, render_template, redirect, url_for, Response
from facenet_pytorch import InceptionResnetV1
import cv2
import numpy as np
from mtcnn.mtcnn import MTCNN
import torch
import pymysql
import mediapipe as mp
from io import BytesIO
import base64

app = Flask(__name__)

# Cargar el modelo InceptionResnetV1
encoder = InceptionResnetV1(pretrained='vggface2', classify=False).eval()

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Conectar a la base de datos MySQL
conn = pymysql.connect(host='localhost', user='root', password='', database='app-recognition')
cursor = conn.cursor()

def generate_frames():
    cap = cv2.VideoCapture(0)  # Usar la cámara web (0 por defecto)

    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        min_detection_confidence=0.5) as face_mesh:

        while True:
            ret, frame = cap.read()

            caras = MTCNN.detect_faces(frame)

            if ret and len(caras) > 0:
                x2, y2, width2, height2 = caras[0]['box']
                face_actual = frame[y2:y2+height2, x2:x2+width2]
                face_actual = cv2.resize(face_actual, (160, 160))
                face_actual = cv2.cvtColor(face_actual, cv2.COLOR_BGR2RGB)
                face_actual = np.transpose(face_actual, (2, 0, 1))
                face_actual = face_actual / 255.0
                face_actual_tensor = torch.tensor(face_actual, dtype=torch.float32)
                face_actual_tensor = face_actual_tensor.unsqueeze(0)

                query = "SELECT nombre_usuario, descriptor_data FROM tabla_descriptores"
                cursor.execute(query)

                nombres_usuarios = []
                descriptors_tensors = []
                for nombre_usuario, descriptor_bytes in cursor.fetchall():
                    descriptor_array = np.frombuffer(descriptor_bytes, dtype=np.float32)
                    descriptor_tensor = torch.tensor(descriptor_array, dtype=torch.float32).unsqueeze(0)
                    nombres_usuarios.append(nombre_usuario)
                    descriptors_tensors.append(descriptor_tensor)

                for nombre_usuario, descriptor_tensor in zip(nombres_usuarios, descriptors_tensors):
                    distancia_euclidiana = torch.norm(descriptor_tensor - face_actual_tensor)
                    umbral_similitud = 0.9
                    if distancia_euclidiana < umbral_similitud:
                        print(f"El rostro capturado coincide con el de {nombre_usuario} en la base de datos.")

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(frame_rgb)

                if results.multi_face_landmarks is not None:
                    for face_landmarks in results.multi_face_landmarks:
                        mp_drawing.draw_landmarks(
                            image=frame,
                            landmark_list=face_landmarks,
                            connections=mp_face_mesh.FACEMESH_TESSELATION,
                            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=1, circle_radius=1),
                            connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=1)
                        )

            ret, buffer = cv2.imencode('.jpg', frame)
            if ret:
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/reconocimiento_facial')
def reconocimiento_facial():
    return render_template('reconocimiento_facial.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Ruta para el formulario
@app.route('/form')
def form():
    return render_template('form.html')

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
    app.run(debug=True, port=3000)
