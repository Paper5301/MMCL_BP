import cv2
import mediapipe as mp
import pickle
path = 'Data/Subject17/Subject17_Jul26_Cam1.avi'



# Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

# Image
cap = cv2.VideoCapture(path)

while True:
    ret, image = cap.read()
    if ret is not True:
        break
    height, width,_ = image.shape
    print(height,width)
    rgb_image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

    # Results of Facial Landmarks
    result = face_mesh.process(rgb_image)
    Landmark = []
    lm = [10,50,67,280,297]
    alpha = 4
    print(image.shape)
    for facial_landmarks in result.multi_face_landmarks:
        for i in lm:
            pt1 = facial_landmarks.landmark[i]
            x = int(pt1.x * width)
            y = int(pt1.y * height)
            cv2.circle(image,(x,y), 3, (100,100,0))
            Landmark.append(image[x-int(alpha/2):x+int(alpha/2),y-int(alpha/2):y+int(alpha/2),:])
    cv2.imshow("Image",image)
    cv2.waitKey(1)


filename = 'landmarks.pkl'

with open(filename, 'wb') as file:
    pickle.dump(Landmark, file)

print(f"List saved to {filename}")