from flask import Flask, render_template, send_file
import sqlite3
import pandas as pd
import os

app = Flask(__name__)

@app.route('/')
def index():
    conn = sqlite3.connect('data/violations.db')
    df = pd.read_sql_query("SELECT * FROM violations ORDER BY timestamp DESC", conn)
    conn.close()
    return render_template('index.html', records=df.to_dict(orient='records'))

@app.route('/download')
def download_csv():
    df = pd.read_sql("SELECT * FROM violations", sqlite3.connect('data/violations.db'))
    path = 'reports/violations_export.csv'
    df.to_csv(path, index=False)
    return send_file(path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
