from flask import Flask,render_template,Response
import cv2
import mediapipe as mp

app=Flask(__name__)
camera=cv2.VideoCapture(0)

mp_drawing = mp.solutions.drawing_utils 
mp_drawing_styles = mp.solutions.drawing_styles
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
mp_face_mesh = mp.solutions.face_mesh

def generate_frames():
    while True:
            
        ## read the camera frame
        success,frame=camera.read()
        if not success:
            break
        else:
            height,width,_=frame.shape
            # print(height)
            # print(width)
            # x=height//2
            # y=width//2
            with mp_face_mesh.FaceMesh(max_num_faces=1,refine_landmarks=True,min_detection_confidence=0.5,min_tracking_confidence=0.5) as face_mesh:
                frame.flags.writeable = False
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(frame)

            # Draw the face mesh annotations on the image.
                frame.flags.writeable = True
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        # mp_drawing.draw_landmarks(
                        #     image=frame,
                        #     landmark_list=face_landmarks,
                        #     connections=mp_face_mesh.FACEMESH_TESSELATION,
                        #     landmark_drawing_spec=None,
                        #     connection_drawing_spec=mp_drawing_styles
                        #     .get_default_face_mesh_tesselation_style())
                        for i in range(0,468,40):
                            pt1=face_landmarks.landmark[i]
                            x=int(pt1.x*width)
                            y=int(pt1.y*height)
                            cv2.circle(frame,(x,y),4,(100,100,0),-1)
                            # time.sleep(4)
                            cv2.putText(frame,str(i),(x,y),1,0.5,(0,0,0))
                else: 
                    continue
            # cv2.putText(frame,str("This is a test image"),(y,x),1,1,(0,0,0))
            ret,buffer=cv2.imencode('.jpg',frame)
            frame=buffer.tobytes()

        yield(b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__=="__main__":
    app.run(debug=True)
