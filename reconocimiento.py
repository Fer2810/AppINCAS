import cv2
import mediapipe as mp
import mysql.connector

# Conexión a la base de datos
db_connection = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    database="app-recognition"
)
db_cursor = db_connection.cursor()

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

with mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=2,
    min_detection_confidence=0.5) as face_mesh:

    while True:
        ret, frame = cap.read()
        if ret == False:
            break
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)

        if results.multi_face_landmarks is not None:
            for face_landmarks in results.multi_face_landmarks:
                landmark_data = []  # Almacena las coordenadas de los puntos
                for landmark in face_landmarks.landmark:
                    landmark_data.extend([landmark.x, landmark.y, landmark.z])

                # Insertar en la base de datos
                insert_query = "INSERT INTO facial_landmarks (x, y, z) VALUES (%s, %s, %s)"
                db_cursor.execute(insert_query, tuple(landmark_data))
                db_connection.commit()

                mp_drawing.draw_landmarks(frame, face_landmarks,
                                          mp_face_mesh.FACEMESH_CONTOURS,
                                          mp_drawing.DrawingSpec(color=(0,255,255), thickness=1, circle_radius=1),
                                          mp_drawing.DrawingSpec(color=(255,0,255), thickness=1))

        cv2.imshow("Frame", frame)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

cap.release()
cv2.destroyAllWindows()
db_cursor.close()
db_connection.close()
