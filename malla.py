import cv2
import mediapipe as mp

mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Inicializa la cámara
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection, mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:

    while True:
        ret, frame = cap.read()
        if ret == False:
            break
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detección de rostro con FaceDetection
        results_detection = face_detection.process(frame_rgb)

        if results_detection.detections:
            for detection in results_detection.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                
                
                 # Aumenta solo la altura (largo) hacia arriba
                scaling_factor_height = 1.2  # Cambia este valor según tus necesidades
                x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih) - int(bboxC.height * ih * (scaling_factor_height - 1)), int(bboxC.width * iw), int(bboxC.height * ih * scaling_factor_height)


                # Dibuja el cuadro del rostro detectado
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Definir una región de interés (ROI) para FaceMesh dentro del cuadro del rostro
                face_roi = frame[y:y+h, x:x+w]
                
                if not face_roi is None:
                    frame_rgb_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
                else:
                    print("Error: The image or frame is empty.")


                # Procesamiento de landmarks faciales en la ROI con FaceMesh
                frame_rgb_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
                results_mesh = face_mesh.process(frame_rgb_roi)

                if results_mesh.multi_face_landmarks:
                    for face_landmarks in results_mesh.multi_face_landmarks:
                        # Dibuja landmarks faciales en la ROI
                        mp_drawing.draw_landmarks(
                            image=face_roi,
                            landmark_list=face_landmarks,
                            connections=mp_face_mesh.FACEMESH_TESSELATION,
                            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=1, circle_radius=1),
                            connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=1)
                        )

        cv2.imshow("Frame", frame)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

cap.release()
cv2.destroyAllWindows()
