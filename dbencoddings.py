from facenet_pytorch import InceptionResnetV1
import cv2
import numpy as np
from mtcnn.mtcnn import MTCNN
import torch
import pymysql

# Cargar el modelo InceptionResnetV1
encoder = InceptionResnetV1(pretrained='vggface2', classify=False).eval()

# Crear una instancia de MTCNN
mtcnn = MTCNN()

# Conectar a la base de datos MySQL
conn = pymysql.connect(host='localhost', user='root', password='', database='app-recognition')
cursor = conn.cursor()

# Crear la tabla si no existe
create_table_query = """
CREATE TABLE IF NOT EXISTS tabla_descriptores (
    descriptor_data LONGBLOB,
    nombre_usuario VARCHAR(255),
    PRIMARY KEY(descriptor_data(767))  # Especifica una longitud de clave de 767 bytes
)
"""
cursor.execute(create_table_query)

# Nombre del usuario
nombre_usuario = "Josue"

# Comprobar si ya existe un descriptor para este usuario
select_query = "SELECT descriptor_data FROM tabla_descriptores WHERE nombre_usuario = %s"
cursor.execute(select_query, (nombre_usuario,))
existing_descriptor = cursor.fetchone()

if existing_descriptor is not None:
    print(f"Ya existe un descriptor facial para el usuario {nombre_usuario}.")
else:
    # Cargar la imagen
    imagen = cv2.imread('img/IMG-20230827-WA0006.jpg')  # Reemplaza con la ubicación de tu imagen
    cara = mtcnn.detect_faces(imagen)

    # Asegúrate de que se haya detectado al menos una cara
    if len(cara) > 0:
        # Extraer la región de la cara
        x, y, width, height = cara[0]['box']
        face = imagen[y:y+height, x:x+width]
        
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
        
        # Almacenar el descriptor facial en la base de datos junto con el nombre de usuario
        insert_query = "INSERT INTO tabla_descriptores (descriptor_data, nombre_usuario) VALUES (%s, %s)"
        cursor.execute(insert_query, (descriptor_bytes, nombre_usuario))
        conn.commit()  # Guardar los cambios en la base de datos
        
        # Ahora tienes el descriptor facial almacenado en la base de datos
        print(f"Descriptor facial almacenado en la base de datos para el usuario {nombre_usuario}.")
    else:
        print("No se detectaron caras en la imagen.")

# Cerrar la conexión a la base de datos
conn.close()
