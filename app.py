from flask import Flask,render_template, request
import cv2
import mediapipe as mp
import base64
import numpy as np

app=Flask(__name__)

mp_drawing = mp.solutions.drawing_utils 
mp_drawing_styles = mp.solutions.drawing_styles
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
mp_face_mesh = mp.solutions.face_mesh



def analyze_frame(base64_jpeg):
    # Decode image from base64
    decoded_data = base64.b64decode(base64_jpeg)
    np_data = np.frombuffer(decoded_data,np.uint8)
    frame = cv2.imdecode(np_data,cv2.IMREAD_UNCHANGED)
    height,width,_=frame.shape
    with mp_face_mesh.FaceMesh(max_num_faces=1,refine_landmarks=True,min_detection_confidence=0.5,min_tracking_confidence=0.5) as face_mesh:
        frame.flags.writeable = False
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame)

        # Draw the face mesh annotations on the image.
        frame.flags.writeable = True
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                for i in range(0,468,40):
                    pt1=face_landmarks.landmark[i]
                    x=int(pt1.x*width)
                    y=int(pt1.y*height)
                    cv2.circle(frame,(x,y),4,(100,100,0),-1)
                    cv2.putText(frame,str(i),(x,y),1,0.5,(0,0,0))
        
    
    ret,buffer=cv2.imencode('.jpg',frame)
    base64_encoded = base64.b64encode(buffer)
    return base64_encoded

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    img = analyze_frame(request.get_data())
    return img

if __name__=="__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
