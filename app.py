from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from flask_socketio import SocketIO
import numpy as np
import json
import glob,os
import subprocess
import socket,pickle
import cv2
from multiprocessing.connection import Client

os.environ['FLASK_ENV'] = 'production'


app = Flask(__name__)
cors = CORS(app)

socketio = SocketIO(app)

@app.route('/send_message', methods=['POST'])
def send_message():
    message = request.json
    socketio.emit('message_to_html', message)  # Emit the message
    # retrieve image
    return "Message received and sent!"

@app.route('/save_fps', methods=['POST'])
def save_fps():
    try:
        data = request.json
        fps = data.get('fps')
        with open('fps.log','a+') as f:
            f.write(fps+'\n')
        return jsonify({"result": f"FPS logged."})
    except Exception as e:
        return str(e), 500
    
@app.route('/save_results', methods=['POST'])
def save_results():
    try:
        images = request.files.getlist('images')
        prefix = request.form['prefix']
        object = request.form['object']
        channel = request.form['channel']
        load_time = request.form['load_time']
        fps = request.form['fps']
        for i, image_data in enumerate(images):
            image_data.save(f'profiling_cache/{prefix}{i+1}.png')
        # only log when measuing the downloading time
        if float(fps) > 0:
            with open('metrics.log','a+') as f:
                f.write(f'{object},{channel},{load_time},{fps}\n')
    except Exception as e:
        return str(e), 500
    
    # Assuming you have some data to send
    data_to_send = {'message': 'Results saved.'}

    address = ('localhost', 8015)
    conn = Client(address, authkey=b'secret password')
    conn.send(pickle.dumps(data_to_send))
    conn.close()

    # # Set up a socket connection to communicate with the other Python program
    # with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    #     s.connect(('localhost', 8015))  # Adjust IP and port as needed
    #     s.sendall(pickle.dumps(data_to_send))

    return 'Results saved successfully', 200

# Define the directory you want to serve
directory_path = './'

@app.route('/<path:filename>')
def serve_file(filename):
    return send_from_directory(directory_path, filename)

def prune(object_name, prune_chan, d):
    # the number of prunable layers is hard-coded to 2 in MobileNeRF
    prunable_num = 2
    channel_imp = [[] for _ in range(prunable_num)]
    if d == 8:
        with open(object_name + '_phone/mlp.json', 'r') as file:
            data = json.load(file)
    else:
        with open(object_name + f'_phone/mlp.{d}.json', 'r') as file:
            data = json.load(file)

    # cal importance
    for obj in data:
        if isinstance(data[obj],list):
            param = np.array(data[obj])
            if obj == '1_weights':
                channel_imp[0] += np.abs(param).sum(axis=-1).tolist()
            elif obj == '2_weights':
                channel_imp[1] += np.abs(param).sum(axis=-1).tolist()

    # prune channels
    for i in range(prunable_num):
        flat_imp = np.array(channel_imp[i])
        sorted_imp = np.sort(flat_imp)
        if prune_chan > len(sorted_imp):
            prune_chan = len(sorted_imp)
        threshold = sorted_imp[-prune_chan]
        in_weight_name = f'{i}_weights'
        out_weight_name = f'{i+1}_weights'
        in_bias_name = f'{i}_bias'
        data[in_weight_name] = np.array(data[in_weight_name])[:,channel_imp[i]>=threshold].tolist()
        data[out_weight_name] = np.array(data[out_weight_name])[channel_imp[i]>=threshold].tolist()
        data[in_bias_name] = np.array(data[in_bias_name])[channel_imp[i]>=threshold].tolist()

    # write to json
    with open(object_name + f'_phone/mlp_p.json', 'wb') as f:
        f.write(json.dumps(data).encode('utf-8'))

    print('Prune to ',object_name + f'_phone/mlp_p.json')

@app.route('/prune_request', methods=['POST'])
def prune_request():
    data = request.json  # Assuming the request contains JSON data

    # Access the array
    prune_chan = int(data.get('channel'))
    object_name = data.get('object_name')
    d = int(data.get('d'))

    prune(object_name,prune_chan, d)

    # compare sizes
    before_size = os.path.getsize(object_name + '_phone/mlp.json')
    after_size = os.path.getsize(object_name + '_phone/mlp_p.json')

    saving = (1-after_size/before_size)*100
    saving_in_MB = (before_size - after_size)/1024

    return jsonify({"result": f"Prune complete. Saved {saving:.2f}%, {saving_in_MB:.2f}KB"})

@app.route('/png_request', methods=['POST'])
def png_request():
    data = request.json  # Assuming the request contains JSON data

    # Access the array
    d = int(data.get('d'))
    f = int(data.get('f'))
    l = int(data.get('l'))
    s = int(data.get('s'))
    c = int(data.get('channel'))
    object_name = data.get('object_name')

    prune(object_name,c,d)

    # texture compression
    pattern = object_name + '_phone/shape[0-9].pngfeat[0-9].png'
    png_files = glob.glob(pattern)
    for file in png_files:
        basename = file[:-4]
        print(file)
        if d == 8:
            subprocess.run(["convert",
                            basename + '.png',
                            '-depth', f'{d}',
                            '-define', f'png:compression-filter={f}',
                            '-define', f'png:compression-level={l}',
                            '-define', f'png:compression-strategy={s}',
                            basename + '.x.png',
                            ], 
                            stdout=subprocess.PIPE, text=True)
        else:
            tmp_file = basename + f'.{d}.png'
            if not os.path.exists(tmp_file):
                img = cv2.imread(file,cv2.IMREAD_UNCHANGED)
                img[:,:,3] = (img[:,:,3] - 128) * 2 # 0-255
                
                mult = 2**(8-d) # 1 (d=8), 128 (d=1)
                valid_pixels = (img[..., 2] != 0)
                img[valid_pixels] = np.clip(img[valid_pixels] // mult * mult, 1, 255)

                img[:,:,3] = img[:,:,3] // 2 + 128
                cv2.imwrite(tmp_file, img, [cv2.IMWRITE_PNG_COMPRESSION,9])
            subprocess.run(["convert",
                            tmp_file,
                            '-depth', f'8',
                            '-define', f'png:compression-filter={f}',
                            '-define', f'png:compression-level={l}',
                            '-define', f'png:compression-strategy={s}',
                            basename + '.x.png',
                            ], 
                            stdout=subprocess.PIPE, text=True)
        
    # check size
    pattern = object_name + '_phone/shape[0-9].pngfeat[0-9].x.png'
    new_files = glob.glob(pattern)
    before_size, after_size = 0,0
    for file in png_files:
        before_size += os.path.getsize (file)

    for file in new_files:
        after_size += os.path.getsize (file)

    saving = (1-after_size/before_size)*100
    saving_in_MB = (before_size - after_size)/1024/1024

    return jsonify({"result": f"PNG compression (d:{d}, f:{f}, l:{l}, s:{s}) complete. Saved: {saving:.2f}%, {saving_in_MB:.2f}MB, final size:{after_size/2**20:.2f}MB."})

@app.route('/draco_request', methods=['POST'])
def draco_request():
    data = request.json  # Assuming the request contains JSON data

    # Access the array
    qp = int(data.get('qp'))
    qt = int(data.get('qt'))
    cl = int(data.get('cl'))
    object_name = data.get('object_name')

    # Use glob to search for files with .obj extension
    obj_files = glob.glob(object_name + f'_phone/' + '*.obj')

    # DRACO
    for file in obj_files:
        basename = file[:-4]
        result = subprocess.run(["draco_encoder",
                                "-i", basename + '.obj',
                                "-o", basename + '.drc',
                                "-qp", f"{qp}",
                                "-qt", f"{qt}",
                                "-cl", f"{cl}",
                                ], 
                                stdout=subprocess.PIPE, text=True)

        # Print the output
        print(result.stdout)

    # check file sizes
    drc_files = glob.glob(object_name + f'_phone/' + '*.drc')
    before_size, after_size = 0,0
    for file in obj_files:
        before_size += os.path.getsize (file)

    for file in drc_files:
        after_size += os.path.getsize (file)

    saving = (1-after_size/before_size)*100
    saving_in_MB = (before_size - after_size)/1024/1024

    return jsonify({"result": f"DRACO compression (qp:{qp}, qt:{qt}, cl:{cl}) complete. Saved {saving:.2f}%, {saving_in_MB:.2f}MB, final size:{after_size/2**20:.2f}MB."})


if __name__ == '__main__':
    app.run(debug=False,host='130.126.139.208',port=8000)
