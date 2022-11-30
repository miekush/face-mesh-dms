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
    {'name': 'Right Eye', 'points': right_eye_points, 'coordinates': [], 'area': 0, 'threshold': 60},
    {'name': 'Left Eye', 'points': left_eye_points, 'coordinates': [], 'area': 0, 'threshold': 60},
    {'name': 'Mouth', 'points': mouth_points, 'coordinates': [], 'area': 0, 'threshold': 100}]

#counters for drowsiness events
eye_closure_counter = 0
blink_counter = 0
microsleep_counter = 0
sleep_event = False
mouth_open_counter = 0
yawn_counter = 0
yawn_event = False

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

    #check for eye closure
    if (features[0]['area'] < features[0]['threshold']) and (features[1]['area'] < features[1]['threshold']):
      #increment eye closure frame counter
      eye_closure_counter += 1
      #print("Eye Closure Count:"+ str(eye_closure_counter))
      #sleep
      if eye_closure_counter > 50:
        sleep_event = True
    #eyes open
    else:
      #blink
      if (eye_closure_counter > 0) and (eye_closure_counter < 5):
        #print("Blink detected!")
        blink_counter += 1
      #microsleep
      elif (eye_closure_counter >= 5) and (eye_closure_counter < 50):
        #print("Microsleep detected!")
        microsleep_counter += 1
      #sleep
      elif sleep_event:
        print("Sleep Duration: " + str(eye_closure_counter))
        #reset flag
        sleep_event = False
      #reset counter
      eye_closure_counter = 0

    #mouth open
    if features[2]['area'] > features[2]['threshold']:
        #increment mouth open frame counter
        mouth_open_counter += 1
        if mouth_open_counter > 50:
          yawn_event = True
    #mouth closed
    else:
        if yawn_event:
            yawn_counter += 1
            print("Yawn Duration: " + str(mouth_open_counter))
        #reset counter
        mouth_open_counter = 0
        #reset flag
        yawn_event = False

    #add blink counter value to image
    image = cv2.putText(
    img = image,
    text = "Blink Count: " + str(blink_counter),
    org = (20, 130),
    fontFace = cv2.FONT_HERSHEY_DUPLEX,
    fontScale = 0.8,
    color = (255, 0, 0),
    thickness = 1
    )

    #add microsleep counter value to image
    image = cv2.putText(
    img = image,
    text = "Microsleep Count: " + str(microsleep_counter),
    org = (20, 160),
    fontFace = cv2.FONT_HERSHEY_DUPLEX,
    fontScale = 0.8,
    color = (255, 0, 0),
    thickness = 1
    )

    #add yawn counter value to image
    image = cv2.putText(
    img = image,
    text = "Yawn Count: " + str(yawn_counter),
    org = (20, 190),
    fontFace = cv2.FONT_HERSHEY_DUPLEX,
    fontScale = 0.8,
    color = (255, 0, 0),
    thickness = 1
    )    

    #add sleep event warning to image
    if sleep_event:
        #add sleep detected warning to image
        image = cv2.putText(
        img = image,
        text = "Sleep Detected!",
        org = (125, 300),
        fontFace = cv2.FONT_HERSHEY_DUPLEX,
        fontScale = 1.5,
        color = (0, 0, 255),
        thickness = 2
        )

    if yawn_event:
        #add yawn detected warnign to image
        image = cv2.putText(
        img = image,
        text = "Yawn Detected!",
        org = (125, 250),
        fontFace = cv2.FONT_HERSHEY_DUPLEX,
        fontScale = 1.5,
        color = (0, 0, 255),
        thickness = 2
        )      

    cv2.imshow('Driver Monitoring System', image)
    if cv2.waitKey(5) & 0xFF == 27:
      break

cap.release()