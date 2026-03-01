import cv2
import numpy as np
import tensorflow as tf
import json
from tensorflow.keras.applications.resnet50 import preprocess_input

with open('class_names.json', 'r') as f:
    labels = json.load(f)

interpreter = tf.lite.Interpreter(model_path='veg.tflite')
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def predict(frame):
    img = cv2.resize(frame, (100, 100))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.expand_dims(img, axis=0).astype(np.float32)
    img = preprocess_input(img)
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    class_index = np.argmax(output[0])
    confidence = output[0][class_index] * 100
    return labels[str(class_index)], confidence

vid = cv2.VideoCapture(2)

while True:
    ret, frame = vid.read()
    if not ret:
        print("Failed to grab frame")
        break

    pred_label, confidence = predict(frame)

    cv2.putText(frame, f"It is a {pred_label} ({confidence:.1f}%)", (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5, (255, 255, 255), 3, 2)

    cv2.imshow('Camera feed', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()