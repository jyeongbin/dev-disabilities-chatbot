from flask import Flask, request, render_template, jsonify
from flask_cors import CORS
import openai
import os 
import tiktoken
from langchain. embeddings.openai import OpenAIEmbeddings 
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain. chains import ChatVectorDBChain 
from langchain.document_loaders import TextLoader
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import OpenAI
# 쓰레드에 필요한 부분
import threading
import queue
# 얼굴인식에 필요한 부분
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

# 감정을 저장할 변수들 초기화
max_emotion = ""
total_frames = 0
max_emotion_frames = 0

# 여기서부터는 챗봇 & flask
# key는 github push를 위해 삭제한 점 양해 부탁드립니다
# os.environ["OPENAI_API_KEY"] = key

# openai.api_key= key

app = Flask(__name__)  # __name__ == '__main__'
CORS(app)

# 감정 인식을 시작할 때 사용할 플래그
condition_to_start_emotion_recognition = False

# 쳇봇 모델을 구축할 때 사용할 플래그
# start_generation = False

loader = TextLoader('./chatbot.txt', encoding='utf-8')
data =loader.load()
print(f'{len(data)}개의 문서')
print(f'문서에 {len(data[0].page_content)}개의 단어를 가지고 있음')

encoding = tiktoken.get_encoding('cl100k_base')
num_tokens = len(encoding.encode(data[0].page_content))
print('num_tokens: ', num_tokens)

# text-embedding-ada-002 입력 가능한 토큰수는 최대 2048
text_splitter = RecursiveCharacterTextSplitter(chunk_size=180, chunk_overlap=0)
docs = text_splitter.split_documents(data)
print(f'{len(docs)}개의 문서 존재')

persist_directory = "./vectordb"
shutil.rmtree(persist_directory, ignore_errors=True) 
os.mkdir(persist_directory)
embeddings = OpenAIEmbeddings()
vectordb = Chroma.from_documents(docs, embeddings, persist_directory=persist_directory)
vectordb.persist()

# model = ChatOpenAI(model_name='gpt-3.5-turbo-16k', temperature=0, max_tokens=200)

chain = ChatVectorDBChain.from_llm(model, vectordb, return_source_documents=True)

# 챗봇 모델 구현 시기를 결정하는 플래그 생성 및 hobby 값 초기화
start_chatbot_generaion = False
hobby = ""
training_data = ""

def Select_training_data(hobby):
    global training_data
    if hobby == "운동":
        training_data = './exercise.txt'
    elif hobby == "장난감":
        training_data = './chatbot.txt'
    elif hobby == "요리":
        training_data = './cook.txt'
    elif hobby == "만화":
        training_data = './tv.txt'
    elif hobby == "애완동물":
        training_data = './animal.txt'
    return training_data


def initialize_chatbot_model(chain_queue):
    if start_chatbot_generaion:
        Select_training_data(hobby)
        loader = TextLoader(training_data, encoding='utf-8')
        data =loader.load()
        print(f'{len(data)}개의 문서')
        print(f'문서에 {len(data[0].page_content)}개의 단어를 가지고 있음')

        encoding = tiktoken.get_encoding('cl100k_base')
        num_tokens = len(encoding.encode(data[0].page_content))
        print('num_tokens: ', num_tokens)

        # text-embedding-ada-002 입력 가능한 토큰수는 최대 2048
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=0)
        docs = text_splitter.split_documents(data)
        print(f'{len(docs)}개의 문서 존재')

        persist_directory = "./vectordb"
        embeddings = OpenAIEmbeddings()
        vectordb = Chroma.from_documents(docs, embeddings, persist_directory=persist_directory)
        vectordb.persist()

        model = ChatOpenAI(model_name='gpt-3.5-turbo-16k', temperature=0, max_tokens=200)

        chain = ChatVectorDBChain.from_llm(model, vectordb, return_source_documents=True)

        # chain 을 chain_queue에 저장
        chain_queue.put(chain)


# 스레드에서 실행할 함수
def run_emotion_recognition(result_queue):
    global perform_emotion_recognition
    global condition_to_start_emotion_recognition
    global max_emotion_frames
    global total_frames
    global max_emotion
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
                break
        
        # 카메라 닫기
        camera.release()
        cv2.destroyAllWindows()
        # 루프를 종료한 후 인식 된 총 프레임과 가장 많이 인식된 감정 출력
        print("Total Frames:", total_frames)
        print("Max Emotion:", max_emotion)
        
        # max_emotion 값을 결과 큐에 추가
        result_queue.put(max_emotion)
        
# 라우팅 설정
      
@app.get("/chatbot_scroll") # http://localhost:5000/chatbot_scroll
def chatbot_scroll():
    return render_template("chatbot_scroll.html")

@app.post("/generation") # http://localhost:5000/generation
def model_generation():
    # like.html에서 선택한 hoby 값을 받고 전역변수에 저장한다.
    global hobby
    hobby = request.form.get('hobby')
    print('-> hobby:', hobby)
    print("요청 들어옴")
    
    # 챗봇 구현 시작
    global start_chatbot_generaion
    start_chatbot_generaion = True
   
     # 스레드 시작
    chain_queue = queue.Queue()
    chain_thread = threading.Thread(target=initialize_chatbot_model, args=(chain_queue,))
    chain_thread.start()
    ("쓰레드 시작")
    
    # 스레드의 실행이 완료될 때까지 기다립니다.
    chain_thread.join()
    
    # 스레드 결과 큐에서 max_emotion 값을 가져옵니다.
    global chain
    chain = chain_queue.get()
    print("스레드 완료")
    
    return jsonify({'message': 'Data received successfully'})

@app.get("/inspect") # http://localhost:5000/inspect, 설문 조사
def inspect_form():
    return render_template("inspect.html")

@app.get("/chatbot") # http://localhost:5000/chatbot, 챗봇 화면
def chatbot_form():
    return render_template("chatbot.html")

@app.get("/btn_chatbot") # http://localhost:5000/btn_chatbot, 버튼 챗봇 화면
def btn_chatbot_form():
    return render_template("btn_chatbot.html")

# chatbot.html 보기 버튼 클릭 시 관심사 선택 창
@app.route('/like')
def like_form():
    return render_template('like.html')

# btnbot.html 보기 버튼 클릭 시 관심사 선택 창
@app.route('/btn_like')
def btn_like_form():
    return render_template('btn_like.html')

# btnbot.html 웃긴 영상 버튼 클릭 시
@app.route('/happy')
def btn_happy_vdo():
    return render_template('happy.html')

# btnbot.html 슬플 때 영상 버튼 클릭 시
@app.route('/sad')
def btn_sad_vdo():
    return render_template('sad.html')

# btnbot.html 화날 때 영상 버튼 클릭 시
@app.route('/angry')
def btn_angry_vdo():
    return render_template('angry.html')

@app.route('/card')
def card_form():
    return render_template('card.html')

@app.route('/bubble pop')
def bubble_form():
    return render_template('bubble pop.html')

@app.route('/num find')
def num_form():
    return render_template('num find.html')

@app.post("/chatbot") # http://localhost:5000/chatbot
def summary_proc():

    question = request.form.get('question')
    print('-> question:', question)
    # return;

    result = chain({'question': question, 'chat_history':[]})
    response = result['answer']
    print('-> response: ', response)

    obj = {
        "response": response
    }
    
    print('-> jsonify(obj)\n', obj)
    return jsonify(obj)

@app.post("/run") # http://localhost:5000/run
def run_video():
    print("요청 받음")
    global condition_to_start_emotion_recognition
    condition_to_start_emotion_recognition = True
    
    question = request.form.get('question')
    print('-> question:', question)
    
    # 스레드 시작
    result_queue = queue.Queue()
    emotion_thread = threading.Thread(target=run_emotion_recognition, args=(result_queue,))
    emotion_thread.start()
    
    # 스레드의 실행이 완료될 때까지 기다립니다.
    emotion_thread.join()
    
    # 스레드 결과 큐에서 max_emotion 값을 가져옵니다.
    max_emotion = result_queue.get()
    
    response = ""  # 초기 response 값을 빈 문자열로 설정
    
    if max_emotion in ["sad", "angry", "neutral"]:
        response = "혹시 오늘 기분 안 좋은 일 있었어?"
    
    obj = {
        "response": response
    }
    
    print('-> jsonify(obj)\n', obj)
    return jsonify(obj)

@app.post("/chatbot_run") # http://localhost:5000/run
def chatbot_run_video():
    print("요청 받음")
    global condition_to_start_emotion_recognition
    condition_to_start_emotion_recognition = True
    
    question = request.form.get('question')
    print('-> question:', question)
    
    # 스레드 시작
    result_queue = queue.Queue()
    emotion_thread = threading.Thread(target=run_emotion_recognition, args=(result_queue,))
    emotion_thread.start()
    
    # 스레드의 실행이 완료될 때까지 기다립니다.
    emotion_thread.join()
    
    # 스레드 결과 큐에서 max_emotion 값을 가져옵니다.
    max_emotion = result_queue.get()
    
    response = ""  # 초기 response 값을 빈 문자열로 설정
    
    if max_emotion in ["sad", "angry", "neutral"]:
        response = "혹시 오늘 기분 안 좋은 일 있었어?"
    
    obj = {
        "response": response
    }
    
    print('-> jsonify(obj)\n', obj)
    return jsonify(obj)

@app.post("/hobby") # http://localhost:5000/chatbot
def hobby_proc():
    
    question = request.form.get('question')
    print('-> question:', question)

    response = ""  # 초기 response 값을 빈 문자열로 설정
    
     #운동
    if  '축구' in question:
        response = " 축구를 좋아하는구나! 축구가 왜 좋아?"

    elif '야구' in question:
        response = " 야구를 좋아하는구나! 야구가 왜 좋아?"
        
    elif '수영' in question:
        response = " 수영을 좋아하는구나! 수영이 왜 좋아?"
    
    elif '걷기' in question:
        response = " 걷는 거를 좋아하는구나! 걷는 게 왜 좋아?"
    
    elif '볼링' in question:
        response = " 볼링을 좋아하는구나! 볼링이 왜 좋아?"
    
    elif '자전거' in question:
        response = " 자전거 타는 거를 좋아하는구나! 자전거 타는 게 왜 좋아?"
        
    # 그림 그리기
    elif '사자' in question:
        response = " 사자를 좋아하는구나! 사자 그림이 왜 좋아?"

    elif '코끼리' in question:
        response = " 코끼리를 좋아하는구나! 코끼리 그림이 왜 좋아?"

    elif '기린' in question:
        response = " 기린을 좋아하는구나! 기린 그림이 왜 좋아?"

    elif '호랑이' in question:
        response = " 호랑이를 좋아하는구나! 호랑이 그림이 왜 좋아?"

    elif '하마' in question:
        response = " 하마를 좋아하는구나! 하마 그림이 왜 좋아?"
        
    elif '고래' in question:
        response = " 고래를 좋아하는구나! 고래 그림이 왜 좋아?"

    elif '상어' in question:
        response = " 상어를 좋아하는구나! 상어 그림이 왜 좋아?"
   
    elif '가족' in question:
        response = " 가족 그리는 거를 좋아하는구나! 좋아하는 이유가 있어?"

    elif '엄마' in question:
        response = " 엄마 그리는 거를 좋아하는구나! 좋아하는 이유가 있어?"

    elif '아빠' in question:
        response = " 아빠 그리는 거를 좋아하는구나! 좋아하는 이유가 있어?"
        
    elif '나무' in question:
        response = " 나무를 좋아하는구나! 나무 그림이 왜 좋아?"
        
    elif '구름' in question:
        response = " 구름을 좋아하는구나! 구름 그림이 왜 좋아?"
        
    elif '달' in question:
        response = " 달을 좋아하는구나! 달 그림이 왜 좋아?"

    # 독서 or 책 읽기
    
    elif '동화' in question:
        response = " 동화를 좋아하는구나! 동화가 왜 좋아?"
        
    elif '소설' in question:
        response = "소설을 좋아하는구나! 소설이 왜 좋아?"
        
    elif '만화' in question:
        response = " 만화를 좋아하는구나! 만화가 왜 좋아?"
        
    elif '우주' in question:
        response = " 우주에 관심이 많구나! 우주가 왜 좋아?"
    
    #노래 부르기
    
    elif '동요' in question:
        response = " 동요 부르는 거를 좋아하는구나! 동요가 왜 좋아?"
        
    elif '클래식' in question:
        response = " 클래식을 좋아하는구나! 클래식이 왜 좋아?"
        
    elif '재즈' in question:
        response = " 재즈를 좋아하는구나! 재즈가 왜 좋아?"
        
    elif '한국' in question:
        response = " 한국노래를 좋아하는구나! 한국노래가 왜 좋아?" 
        
    # 장난감
    
    elif '레고' in question:
        response = " 레고 좋아하는구나! 레고가 왜 좋아?"
        
    elif '인형' in question:
        response = " 인형을 좋아하는구나! 인형이 왜 좋아?"
        
    elif '피규어' in question:
        response = " 피규어를 좋아하는구나! 피규어가 왜 좋아?"
        
    #애완동물
    elif '개' in question:
        response = " 개를 좋아하는구나! 개가 왜 좋아?"
        
    elif '강아지' in question:
        response = " 강아지를 좋아하는구나! 강아지가 왜 좋아?"
        
    elif '고양이' in question:
        response = " 고양이를 좋아하는구나! 고양이가 왜 좋아?"
        
    elif '햄스터' in question:
        response = " 햄스터를 좋아하는구나! 햄스터가 왜 좋아?"
        
    elif '토끼' in question:
        response = " 토끼를 좋아하는구나! 토끼가 왜 좋아?"
        
    elif '앵무새' in question:
        response = " 앵무새를 좋아하는구나! 앵무새가 왜 좋아?"
        
    elif '잉꼬' in question:
        response = " 잉꼬를 좋아하는구나! 잉꼬가 왜 좋아?"
        
    elif '이구아나' in question:
        response = " 이구아나를 좋아하는구나! 이구아나가 왜 좋아?"

    elif '거북이' in question:
        response = " 거북이를 좋아하는구나! 거북이가 왜 좋아?"
        
    elif '조류' in question:
        response = " 조류를 좋아하는구나! 조류가 왜 좋아?"

    elif '물고기' in question:
        response = " 물고기를 좋아하는구나! 물고기가 왜 좋아?"
        
    elif '사슴벌레' in question:
        response = " 사슴벌레를 좋아하는구나! 사슴벌레가 왜 좋아?"
        
    elif '장수풍뎅이' in question:
        response = " 장수풍데이를 좋아하는구나! 장수풍뎅이가 왜 좋아?"
    
    # 만화 보기
    elif '또봇 보기' in question:
        response = " 또봇 보는 거를 좋아하는구나! 또봇이 왜 좋아?"
        
    elif '헬로카봇 보기' in question:
        response = " 헬로카봇 보는 거를 좋아하는구나! 헬로카봇이 왜 좋아?"
        
    elif '로보카폴리 보기' in question:
        response = " 로보카폴리를 좋아하는구나! 로보카폴리가 왜 좋아?"
        
    elif '톰' in question:
         response = " 톰을 좋아하는구나! 톰이 왜 좋아?"
        
    elif '짱구' in question:
         response = " 짱구를 좋아하는구나! 짱구가 왜 좋아?"
        
    #요리
    elif '샌드위치' in question:
        response = " 샌드위치 만드는 거를 좋아하는구나! 샌드위치를 좋아하는 이유가 있어?"
    
    elif '김밥' in question:
        response = " 김밥 만드는 거를 좋아하는구나! 김밥을 좋아하는 이유가 있어?"
        
    elif '빵' in question:
        response = " 빵 만드는 거를 좋아하는구나! 빵을 좋아하는 이유가 있어?"
        
    elif '쿠키' in question:
        response = " 쿠키 만드는 거를 좋아하는구나! 쿠키를 좋아하는 이유가 있어?"
        
    elif '피자' in question:
        response = " 피자 만드는 거를 좋아하는구나! 피자를 좋아하는 이유가 있어?"
    
    obj = {
        "response": response
    }
    
    print('-> jsonify(obj)\n', obj)
    return jsonify(obj)

    
app.run(host="0.0.0.0", port=5000)  # 0.0.0.0: 모든 Host 에서 접속 가능
