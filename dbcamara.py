from facenet_pytorch import InceptionResnetV1
import cv2
import numpy as np
from mtcnn.mtcnn import MTCNN
import torch
import pymysql
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Cargar el modelo InceptionResnetV1
encoder = InceptionResnetV1(pretrained='vggface2', classify=False).eval()

# Crear una instancia de MTCNN
mtcnn = MTCNN()

# Conectar a la base de datos MySQL
conn = pymysql.connect(host='localhost', user='root', password='', database='app-recognition')
cursor = conn.cursor()

# Iniciar la cámara web
cap = cv2.VideoCapture(0)  # Usar la cámara web (0 por defecto)

with mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    min_detection_confidence=0.5) as face_mesh:

    while True:
        # Capturar un fotograma desde la cámara
        ret, frame = cap.read()

        # Detección de caras con MTCNN en el fotograma capturado
        caras = mtcnn.detect_faces(frame)

        if ret and len(caras) > 0:
            # Extraer la región de la cara en el fotograma
            x2, y2, width2, height2 = caras[0]['box']
            face_actual = frame[y2:y2+height2, x2:x2+width2]

            # Redimensionar la cara capturada a 160x160
            face_actual = cv2.resize(face_actual, (160, 160))

            # Preprocesar la imagen capturada para el modelo InceptionResnetV1
            face_actual = cv2.cvtColor(face_actual, cv2.COLOR_BGR2RGB)
            face_actual = np.transpose(face_actual, (2, 0, 1))
            face_actual = face_actual / 255.0
            face_actual_tensor = torch.tensor(face_actual, dtype=torch.float32)
            face_actual_tensor = face_actual_tensor.unsqueeze(0)

            # Ajustar la forma de face_actual_tensor para que tenga una dimensión de 512
            face_actual_tensor = encoder(face_actual_tensor)

            # Consultar los descriptores faciales y nombres de usuario almacenados en la base de datos
            query = "SELECT nombre_usuario, descriptor_data FROM tabla_descriptores"
            cursor.execute(query)

            # Recuperar los descriptores faciales y nombres de usuario como bytes
            nombres_usuarios = []
            descriptors_tensors = []
            for nombre_usuario, descriptor_bytes in cursor.fetchall():
                descriptor_array = np.frombuffer(descriptor_bytes, dtype=np.float32)
                descriptor_tensor = torch.tensor(descriptor_array, dtype=torch.float32).unsqueeze(0)
                nombres_usuarios.append(nombre_usuario)
                descriptors_tensors.append(descriptor_tensor)

            # Comparar los descriptores faciales con el obtenido en tiempo real
            for nombre_usuario, descriptor_tensor in zip(nombres_usuarios, descriptors_tensors):
                # Calcula la distancia euclidiana entre los descriptores (aquí debes implementar tu lógica de comparación)
                distancia_euclidiana = torch.norm(descriptor_tensor - face_actual_tensor)

                # Establecer un umbral de similitud
                umbral_similitud = 0.9  # Puedes ajustar este valor según tus necesidades

                # Comparar la distancia con el umbral
                if distancia_euclidiana < umbral_similitud:
                    print(f"El rostro capturado coincide con el de {nombre_usuario} en la base de datos.")
                    # Puedes agregar aquí acciones adicionales si se reconoce un rostro
                    # Por ejemplo, realizar alguna acción sin mostrar el nombre en la imagen

            # Superponer la malla facial de Mediapipe
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

        # Mostrar el fotograma con la detección de caras y la malla facial
        cv2.imshow('Face Recognition', frame)

        # Salir del bucle si se presiona la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Liberar la cámara y cerrar la ventana
cap.release()
cv2.destroyAllWindows()

# Cerrar la conexión a la base de datos
conn.close()
