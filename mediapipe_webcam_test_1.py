import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh
from shapely.geometry import Polygon

# For webcam input:
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
cap = cv2.VideoCapture(0)

#landmark points for right eye aperture
right_eye_points = [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7]
#landmark points for left eye aperture
left_eye_points = [362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382]
#landmark points for mouth aperture
mouth_points = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]

#list to store all features of interest
features = [
    {'name': 'Right Eye', 'points': right_eye_points, 'coordinates': [], 'area': 0},
    {'name': 'Left Eye', 'points': left_eye_points, 'coordinates': [], 'area': 0},
    {'name': 'Mouth', 'points': mouth_points, 'coordinates': [], 'area': 0}]

with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:
  
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image)

    if not results.multi_face_landmarks:
      continue    

    for face in results.multi_face_landmarks:
        for feature in features:
            feature['coordinates'] = []
            feature['area'] = 0
            for landmark_point in feature['points']:
                #get image size info
                shape = image.shape
                image_h = shape[0]
                image_w = shape[1]
                #extract landmark coordinate and convert using image size
                x = int(face.landmark[landmark_point].x * image_w)
                y = int(face.landmark[landmark_point].y * image_h)
                feature['coordinates'].append([x, y])
                #draw circle
                cv2.circle(image, (x, y), radius=1, color=(0, 0, 255), thickness=1)
            feature['area'] = Polygon(feature['coordinates']).area
    
    #convert the image back to BGR
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    #flip the image horizontally for a selfie-view display.
    image = cv2.flip(image, 1)

    #add feature area values to image
    y=40
    for feature in features:
      image = cv2.putText(
        img = image,
        text = feature['name'] + ": " + str(feature['area']),
        org = (20, y),
        fontFace = cv2.FONT_HERSHEY_DUPLEX,
        fontScale = 0.8,
        color = (255, 0, 0),
        thickness = 1
      )
      y+=30    
    
    cv2.imshow('Driver Monitoring System', image)
    if cv2.waitKey(5) & 0xFF == 27:
      break

cap.release()
