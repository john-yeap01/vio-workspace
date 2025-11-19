from flask import Flask, render_template

app = Flask(__name__)

@app.route("/<name>")
def hello(John):
    return render_template("index.html", name=name)



# $ flask --app flask_test run
# and then open 
# http://127.0.0.1:5000/John