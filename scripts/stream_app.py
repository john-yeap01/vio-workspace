from flask import Flask, Response, render_template_string
import cv2

app = Flask(__name__)

def frames():
    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)  # macOS; drop second arg on Linux/Windows
    if not cap.isOpened():
        raise RuntimeError("Camera not accessible")

    try:
        # Optional: shrink for bandwidth
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if not ok:
                continue
            jpg = buf.tobytes()
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n")
    finally:
        cap.release()

@app.route("/")
def index():
    # tiny inline template to keep it one-file
    return render_template_string("""
      <!doctype html>
      <html><body style="margin:0;background:#111;display:grid;place-items:center;height:100vh">
        <img src="/video" style="max-width:100%;height:auto;border-radius:12px"/>
      </body></html>""")

@app.route("/video")
def video():
    return Response(frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, threaded=True, debug=True)
