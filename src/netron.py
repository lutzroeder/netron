#!/usr/bin/python

import codecs
import os
import platform
import sys
import threading
import webbrowser
from .onnx_ml_pb2 import ModelProto

if sys.version_info[0] > 2:
    from urllib.parse import urlparse
    from http.server import HTTPServer
    from http.server import BaseHTTPRequestHandler
else:
    from urlparse import urlparse
    from BaseHTTPServer import HTTPServer
    from BaseHTTPServer import BaseHTTPRequestHandler

class MyHTTPRequestHandler(BaseHTTPRequestHandler):
    def handler(self):
        if not hasattr(self, 'mime_types_map'):
            self.mime_types_map = {
                '.html': 'text/html',
                '.js':   'text/javascript',
                '.css':  'text/css',
                '.png':  'image/png',
                '.gif':  'image/gif',
                '.jpg':  'image/jpeg',
                '.ico':  'image/x-icon',
                '.json': 'application/json',
                '.pb': 'application/octet-stream',
                '.ttf': 'font/truetype',
                '.woff': 'font/woff',
                '.otf': 'font/opentype',
                '.eot': 'application/vnd.ms-fontobject',
                '.woff': 'application/font-woff',
                '.woff2': 'application/font-woff2',
                '.svg': 'image/svg+xml'
            }
        pathname = urlparse(self.path).path
        folder = os.path.dirname(os.path.realpath(__file__))
        location = folder + pathname;
        status_code = 0
        headers = {}
        buffer = None
        if status_code == 0:
            if pathname == '/':
                with codecs.open(location + 'view-browser.html', mode="r", encoding="utf-8") as open_file:
                    buffer = open_file.read()
                buffer = buffer.replace('{{{title}}}', self.data.file)
                buffer = buffer.encode('utf-8');
                headers['Content-Type'] = 'text/html'
                headers['Content-Length'] = len(buffer)
                status_code = 200
            elif pathname == '/data':
                buffer = self.data.data
                headers['Content-Type'] = 'application/octet-stream'
                headers['Content-Length'] = len(buffer)
                status_code = 200
            else:
                if os.path.exists(location) and not os.path.isdir(location):
                    extension = os.path.splitext(location)[1]
                    content_type = self.mime_types_map[extension]
                    if content_type:
                        with open(location, 'rb') as binary:
                            buffer = binary.read()
                        headers['Content-Type'] = content_type
                        headers['Content-Length'] = len(buffer)
                        status_code = 200
                else:
                    status_code = 404
        if self.verbose:
            print(str(status_code) + ' ' + self.command + ' ' + self.path)
        sys.stdout.flush()
        self.send_response(status_code)
        for key in headers:
            self.send_header(key, headers[key])
        self.end_headers()
        if self.command != 'HEAD':
            if status_code == 404 and buffer is None:
                self.wfile.write(str(status_code))
            elif (status_code == 200 or status_code == 404) and buffer != None:
                self.wfile.write(buffer)
        return
    def do_GET(self):
        self.handler()
    def do_HEAD(self):
        self.handler()
    def log_message(self, format, *args):
        return

class MyHTTPServer(HTTPServer):
    def initialize_data(self, data,verbose):
        self.RequestHandlerClass.data = data
        self.RequestHandlerClass.verbose = verbose

def optimize_onnx(model):
    def remove_tensor_data(tensor):
        del tensor.string_data[:]
        del tensor.int32_data[:]
        del tensor.int64_data[:]
        del tensor.float_data[:]
        tensor.raw_data = None
    # Remove raw initializer data
    onnx_model = ModelProto()
    try:
        onnx_model.ParseFromString(model.data)
    except:
        return False
    for initializer in onnx_model.graph.initializer:
        remove_tensor_data(initializer)
    for node in onnx_model.graph.node:
        for attribute in node.attribute:
            if attribute.HasField('t'):
                remove_tensor_data(attribute.t)
    model.data = onnx_model.SerializeToString()
    return True

def optimize_tf(model):
    return True;

def optimize_tflite(model):
    return True;

class Model:
    def __init__(self, data, file):
        self.data = data
        self.file = file

def serve_data(data, file, verbose=False, browse=False, port=8080, host='localhost', tensor=False):
    server = MyHTTPServer((host, port), MyHTTPRequestHandler)
    model = Model(data, file)
    if not tensor:
        print("Processing '" + file + "'...")
        ok = False
        if not ok and file.endswith('.tflite'):
             ok = optimize_tflite(model)
        if not ok and os.path.basename(file) == 'saved_model.pb':
            ok = optimize_tf(model)
        if not ok and file.endswith('.onnx') or file.endswith('.pb'):
            ok = optimize_onnx(model)
        if not ok and file.endswith('.pb'):
            ok = optimize_tf(model)
    url = 'http://' + host + ':' + str(port)
    print("Serving '" + file + "' at " + url + "...")
    server.initialize_data(model, verbose)
    sys.stdout.flush()
    if browse:
        threading.Timer(1, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    except (KeyboardInterrupt, SystemExit):
        print("\nStopping...")
        server.server_close()

def serve_file(file, verbose=False, browse=False, port=8080, host='localhost', tensor=False):
    print("Reading '" + file + "'...")
    data = None
    with open(file, 'rb') as binary:
        data = binary.read()
    serve_data(data, file, verbose=verbose, browse=browse, port=port, host=host, tensor=tensor)
