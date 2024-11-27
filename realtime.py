# 쓰레드에 필요한 부분
import threading
import queue
#얼굴인식에 필요한 부분
from keras.preprocessing.image import img_to_array
import imutils
import cv2
from keras.models import load_model
import numpy as np
import time
import shutil            

# 얼굴인식과 감정인식 모델 경로를 각 변수에 저장
detection_model_path = 'haarcascade_files/haarcascade_frontalface_default.xml'
emotion_model_path = 'models/_mini_XCEPTION.102-0.66.hdf5'  

# 모델 로드, 감정 리스트 정의
face_detection = cv2.CascadeClassifier(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)
EMOTIONS = ["angry", "disgust", "scared", "happy", "sad", "surprised", "neutral"]

# 감정 인식 수행 시기를 결정하는 플래그 생성
perform_emotion_recognition = False

# 감정 인식을 시작할 때 사용할 플래그
condition_to_start_emotion_recognition = False

# 감정을 저장할 변수들 초기화
max_emotion = ""
total_frames = 0
max_emotion_frames = 0

# 스레드에서 실행할 함수
def run_emotion_recognition(result_queue):
    global perform_emotion_recognition
    global condition_to_start_emotion_recognition
    global max_emotion_frames
    global total_frames
    # condition_to_start_emotion_recognition이 True일 때 감정 인식을 시작
    if condition_to_start_emotion_recognition:
        # 비디오 스트리밍 시작
        cv2.namedWindow('your_face')
        camera = cv2.VideoCapture(0)
    # 감정 인식 시작 시간을 저장하는 변수 초기화
    emotion_recognition_start_time = time.time()

    while True:
        frame = camera.read()[1]
        frame = imutils.resize(frame, width=300)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detection.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),
                                                flags=cv2.CASCADE_SCALE_IMAGE)
        
        canvas = np.zeros((250, 300, 3), dtype="uint8")
        frameClone = frame.copy()
        
        # 2초 대기 후에 perform_emotion_recognition을 True로 설정
        if time.time() - emotion_recognition_start_time >= 2:
            perform_emotion_recognition = True
            print("2초 됌")

        if len(faces) > 0 and perform_emotion_recognition:
            print("감정인식 시작")
            faces = sorted(faces, reverse=True,
                        key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
            (fX, fY, fW, fH) = faces

            roi = gray[fY:fY + fH, fX:fX + fW]
            roi = cv2.resize(roi, (64, 64))
            roi = roi.astype("float") / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            preds = emotion_classifier.predict(roi)[0]
            emotion_probability = np.max(preds)
            label = EMOTIONS[preds.argmax()]

            # 현재 프레임에 대해 확률이 가장 높은 감정을 저장합니다.
            if emotion_probability > max_emotion_frames:
                max_emotion_frames = emotion_probability
                max_emotion = label

            total_frames += 1

            for (i, (emotion, prob)) in enumerate(zip(EMOTIONS, preds)):
                text = "{}: {:.2f}%".format(emotion, prob * 100)
                w = int(prob * 300)
                cv2.rectangle(canvas, (7, (i * 35) + 5), (w, (i * 35) + 35), (0, 0, 255), -1)
                cv2.putText(canvas, text, (10, (i * 35) + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)
                cv2.putText(frameClone, label, (fX, fY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
                cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH), (0, 0, 255), 2)

        cv2.imshow('your_face', frameClone)
        cv2.imshow("Probabilities", canvas)
        
        # Check for key press to toggle emotion recognition
        key = cv2.waitKey(1)

        # 감정인식 시작 후 5초가 지났는지 확인
        if perform_emotion_recognition and time.time() - emotion_recognition_start_time >= 7:
            perform_emotion_recognition = False
            # 5초 대기
            time.sleep(5)
            # 감정 인식 시작 시간을 다시 설정
            emotion_recognition_start_time = time.time()
