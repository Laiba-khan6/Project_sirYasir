import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os

# Function to load the dataset of faces
def load_dataset(data_folder):
    X = []
    y = []

    for subdir in os.listdir(data_folder):
        subdir_path = os.path.join(data_folder, subdir)
        if os.path.isdir(subdir_path):
            for filename in os.listdir(subdir_path):
                if filename.endswith(".jpg"):
                    img_path = os.path.join(subdir_path, filename)
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    X.append(img)
                    y.append(subdir)

    return np.array(X), np.array(y)

# Function to train a face recognition model
# Function to train a face recognition model
def train_face_recognition_model(X, y):
    # Convert labels to numerical values
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Split the dataset into training and testing sets with a smaller test size (e.g., 10%)
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.1, random_state=42)

    # Create an SVM classifier
    clf = SVC(C=1.0, kernel='linear', probability=True, random_state=42)

    # Train the classifier
    clf.fit(X_train, y_train)

    # Predict labels on the test set
    y_pred = clf.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")

    return clf, label_encoder


# Function to recognize faces in a live video stream
def recognize_faces(model, label_encoder):
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            face_roi = gray[y:y+h, x:x+w]

            # Resize the face image to a fixed size (e.g., 100x100 pixels)
            face_roi = cv2.resize(face_roi, (100, 100))

            # Recognize the face
            label_id = model.predict([face_roi.flatten()])[0]
            label = label_encoder.inverse_transform([label_id])[0]

            # Draw a rectangle around the detected face and label it
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow('Face Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Specify the path to the dataset folder containing labeled face images
    data_folder = "C:/Users/sharj/OneDrive/Desktop/data/person"

    # Load the dataset
    X, y = load_dataset(data_folder)

    # Train the face recognition model
    model, label_encoder = train_face_recognition_model(X, y)

    # Recognize faces in the live video stream
    recognize_faces(model, label_encoder)
    
