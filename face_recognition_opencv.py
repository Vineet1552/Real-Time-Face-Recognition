import face_recognition
import cv2
import numpy as np
import os

video_capture = cv2.VideoCapture(0)

# Start superimposing the initial path, the combined operating path to be identified

image1 = face_recognition.load_image_file(os.path.abspath("Robert_downey_jr.jpg"))  #This method returns a normalized version of the pathname path.
image1_face_encoding = face_recognition.face_encodings(image1)[0]                   #return the 128-dimension face encoding for each face in the image.

image2 = face_recognition.load_image_file(os.path.abspath("Johnny_Depp.jpg"))
image2_face_encoding = face_recognition.face_encodings(image2)[0]

image3 = face_recognition.load_image_file(os.path.abspath("Messi.jpg"))
image3_face_encoding = face_recognition.face_encodings(image3)[0]



#List of encoded faces from Known faces

known_face_encodings = [
    image1_face_encoding,
    image2_face_encoding,
    image3_face_encoding
]

# list of Known faces
known_face_names = [
    "RDJ",
    "Johnny Depp",
    "lionel Messi"
]


face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

while True:
    ret, frame = video_capture.read()                         #Single frame video
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25) # Resize frame of video to 1/4 size for faster face recognition processing


    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]
    print(rgb_small_frame)


    if process_this_frame:

        # returns a dict of face feature locations (eyes, nose, etc)
        face_locations = face_recognition.face_locations(rgb_small_frame)


        # list of face encodings, compare them to a known face encoding and get a euclidean distance for each comparison face
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See that the face is a match for the known faces.
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)
            name = "Unknown"

            # If a match was found in known_face_encodings, just use the first one.
            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
            face_names.append(name)

    process_this_frame = not process_this_frame
    print("Face detected -- {}".format(face_names))

    # Display the result
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top =top*4
        right *= 4
        bottom *= 4
        left *= 4
        # Draw a box around the face
        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.rectangle(frame, (left, bottom - 25), (right, bottom), (0, 220, 0), cv2.FILLED)
        #(25 --->35,220---->255)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.6, (255, 255, 255), 1)

    # Display the image
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()