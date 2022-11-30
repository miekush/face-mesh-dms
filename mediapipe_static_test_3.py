import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

# For static images:
IMAGE_FILES = ['test.jpg']
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

#landmark points for right eye aperture
right_eye = [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7]
#landmark points for left eye aperture
left_eye = [362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382]
#landmark points for mouth aperture
mouth = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]

#list to store all features of interest
features = [right_eye, left_eye, mouth]

with mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5) as face_mesh:
  
  for idx, file in enumerate(IMAGE_FILES):
    image = cv2.imread(file)
    # Convert the BGR image to RGB before processing.
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    # Print and draw face mesh landmarks on the image.
    
    if not results.multi_face_landmarks:
      continue
    
    annotated_image = image.copy()

    for face in results.multi_face_landmarks:
        for feature in features:
            for landmark_point in feature:
                #get image size info
                shape = annotated_image.shape
                image_h = shape[0]
                image_w = shape[1]
                #extract landmark coordinate and convert using image size
                x = int(face.landmark[landmark_point].x * image_w)
                y = int(face.landmark[landmark_point].y * image_h)
                #draw circle
                cv2.circle(annotated_image, (x, y), radius=5, color=(255, 0, 0), thickness=5)        
    cv2.imwrite('annotated_image' + str(idx) + '.jpg', annotated_image)