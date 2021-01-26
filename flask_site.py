import numpy as np
import json
import helper
import pandas as pd
from flask import Flask, render_template, url_for
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

@app.route('/')
@app.route('/<name>')
def hello_world(name=None):
    return render_template('index.html', graph=url_for('static', filename='test1.png'), name=name)