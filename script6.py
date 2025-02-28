import cv2
import face_recognition
import numpy as np
import os


known_faces_dir = "/Users/vijaysharma/known_faces"


known_faces_info = {
    "/Users/vijaysharma/known_faces/WhatsApp Image 2023-08-30 at 12.56.49 PM (3).jpeg": ("Vijay Sharma", "2203031240435"),
    "/Users/vijaysharma/known_faces/Screenshot 2025-02-24 at 11.25.47 PM.png": ("Hitiesh", "2203031240445")
}


known_face_encodings = []
known_face_names = []


for face_path, (name, enrollment) in known_faces_info.items():
    if os.path.exists(face_path):
        image = face_recognition.load_image_file(face_path)
        encoding = face_recognition.face_encodings(image)

        if encoding:  # Ensure encoding was successful
            known_face_encodings.append(encoding[0])
            known_face_names.append(f"{name}\n{enrollment}")  
        else:
            print(f"⚠️ Warning: No face found in {face_path}")
    else:
        print(f"❌ Error: File not found - {face_path}")


video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances) if face_distances.size > 0 else None

        if best_match_index is not None and matches[best_match_index]:
            name = known_face_names[best_match_index]  

        
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        for i, line in enumerate(name.split("\n")):  
            cv2.putText(frame, line, (left, top - 10 - (i * 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    
    cv2.imshow("Face Recognition", frame)

    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


video_capture.release()
cv2.destroyAllWindows()
