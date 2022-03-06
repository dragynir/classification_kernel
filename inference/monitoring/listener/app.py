from flask import Flask, jsonify, request
import os
import threading
import configparser
import csv


flask_app = Flask(__name__)
WRITER_LOCK = threading.Lock()


metrics_file = './data/birdsy_records.csv'
write_headers = True
if os.path.exists(metrics_file):
    write_headers = False

fieldnames = ['image_url', 'metrics']
csvfile = open(metrics_file, 'a')
writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=';')

if write_headers:
    writer.writeheader()

def write_record(json_data):
    with WRITER_LOCK:
        print('Writing:', json_data) 
        writer.writerow(json_data)
        csvfile.flush()

@flask_app.route("/log", methods=['POST'])
def log_records():

    json_data = request.get_json()

    write_record(json_data)

    return jsonify({"Status": 'ok'}), 200

if __name__ == '__main__':
    port = 9097
    host = '0.0.0.0'
    print(f"Starting server {host}:{port}")
    flask_app.run(debug=False, host=host, port=port, threaded=True)
    csvfile.close()