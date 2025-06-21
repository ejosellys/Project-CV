from flask import Flask, render_template, Response, request, redirect, url_for
import cv2
import os
from ultralytics import YOLO
from werkzeug.utils import secure_filename

app = Flask(__name__)

model = YOLO("weights/best.pt")
camera = cv2.VideoCapture(0)

UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def gen_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            results = model.predict(source=frame, conf=0.5, save=False)
            annotated_frame = results[0].plot()

            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/live')
def live():
    return render_template('live.html')

@app.route('/video')
def video():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return redirect(url_for('index'))

    file = request.files['image']
    if file.filename == '':
        return redirect(url_for('index'))

    filename = secure_filename(file.filename)
    original_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(original_path)

    results = model.predict(source=original_path, conf=0.5, save=False)
    annotated = results[0].plot()

    predicted_filename = 'pred_' + filename
    predicted_path = os.path.join(app.config['UPLOAD_FOLDER'], predicted_filename)
    cv2.imwrite(predicted_path, annotated)

    return render_template('result.html',
                           original=filename,
                           predicted=predicted_filename)

@app.route('/upload_video', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return redirect(url_for('index'))

    file = request.files['video']
    if file.filename == '':
        return redirect(url_for('index'))

    filename = secure_filename(file.filename)
    input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(input_path)

    output_filename = 'pred_' + filename
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)

    # Proses video dengan YOLO
    results = model.predict(
        source=input_path,
        save=True,
        save_txt=False,
        save_conf=False,
        conf=0.5,
        stream=False,
        vid_stride=1
    )

    # ambil file hasil
    save_dir = results[0].save_dir  # lokasi direktori hasil
    for f in os.listdir(save_dir):
        if f.endswith(".mp4"):
            os.rename(os.path.join(save_dir, f), output_path)
            break

    return render_template('result_video.html', video_filename=output_filename)

if __name__ == '__main__':
    app.run(debug=True)
