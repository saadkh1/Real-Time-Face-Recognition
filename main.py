# Import necessary libraries
import cv2
import face_recognition
import os
import glob

# Open the video capture object and start the camera
vc = cv2.VideoCapture(0)

# Set the address of the IP camera to connect to
address = 'http://192.168.1.12:8080/video'
vc.open(address)

# Initialize empty lists for known faces, names, and paths
known_faces = []
known_names = []
known_faces_paths = []

# Set the directory path for registered faces
registered_path = 'registered/'

# Loop through each name in the registered directory and add their images and names to the known_faces_paths and known_names lists
for name in os.listdir(registered_path):
    images_mask = '%s%s/*.jpg' % (registered_path, name)
    images_paths = glob.glob(images_mask) 
    known_faces_paths += images_paths
    known_names += [name for x in images_paths]

# Define a function to get the encodings of a given image path using face_recognition
def get_encodings(img_path):
    load_image = face_recognition.load_image_file(img_path)
    encoding = face_recognition.face_encodings(load_image)
    return encoding[0]

# Get the encodings of all known faces from the known_faces_paths list
known_faces = [get_encodings(img_path) for img_path in known_faces_paths]

# Start a loop to continuously read frames from the video capture object and process them
while True:
    # Read a frame from the video capture object
    ret, frame = vc.read()
    if not ret:
        break
    
    # Convert the frame from BGR to RGB color space
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Use face_recognition to find the location of all faces in the frame
    faces = face_recognition.face_locations(frame_rgb)
    
    # Loop through each face found and draw a rectangle around it, and identify the face using face_recognition compare_faces function
    for face in faces: 
        top, right, bottom, left = face
        cv2.rectangle(frame, (left, top), (right, bottom),(0,0,255), 2)
        face_encoding = face_recognition.face_encodings(frame_rgb, [face])[0]

        results = face_recognition.compare_faces(known_faces, face_encoding, tolerance=0.6)
        if any(results):
            name = known_names[results.index(True)]
        else:
            name = 'unknown'
        
        # Write the name of the identified face on the frame
        cv2.putText(frame, name, (left, bottom + 20), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255), 2)

    # Show the processed frame
    cv2.imshow('Register Face', frame)
    k = cv2.waitKey(1)
    
    # Exit the program if the 'q' key is pressed
    if ord('q') == k:
        break

# Close all windows and release the video capture object
cv2.destroyAllWindows()
vc.release()
