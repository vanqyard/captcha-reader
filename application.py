from flask import Flask  
from flask import render_template
from flask import request

import requests
import lxml.etree
import lxml.html

import read_bmp as captcha_reader

# creates a Flask application, named app
app = Flask(__name__)

def download_captcha():
    url = r'https://ebok.poog.pgnig.pl/login.php'
    data = requests.get(url).content
    root = lxml.html.fromstring(data)
    text = root.xpath("//img[@class='token']/@src")[0]
    name = root.xpath("//input[@id='token']/@value")[0]
    fname = name[:8].replace("/", "x")
    print(name)
    print(fname)

    captcha = "https://ebok.poog.pgnig.pl/" + text
    pic = requests.get(captcha).content

    f = open("./static/" + fname + ".png", 'wb')
    f.write(pic)
    f.close()

    return fname

# a route where we will display a welcome message via an HTML template
@app.route("/")
def hello():
	message = "Hello, World"
	fname = download_captcha()
	captcha_reader.process_image("./static/" + fname + ".png", "./static/" + fname + "_")
	return render_template('index.html', message=message, captcha=fname)

# run the application
if __name__ == "__main__":  
	app.run(debug=True)

