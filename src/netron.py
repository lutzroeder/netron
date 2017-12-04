#!/usr/bin/python

import codecs
import os
import platform
import sys
import webbrowser
import onnx_ml_pb2

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
                buffer = buffer.replace('{{{title}}}', self.model.file)
                headers['Content-Type'] = 'text/html'
                headers['Content-Length'] = len(buffer)
                status_code = 200
            elif pathname == '/model':
                buffer = self.model.data
                headers['Content-Type'] = 'text/plain'
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
            if status_code == 404 and buffer == None:
                self.wfile.write(str(status_code))
            elif (status_code == 200 or status_code == 404) and buffer != None:
                self.wfile.write(buffer)
        return;
    def do_GET(self):
        self.handler();
    def do_HEAD(self):
        self.handler();
    def log_message(self, format, *args):
        return

class MyHTTPServer(HTTPServer):
    def serve_forever(self, model, verbose):
        self.RequestHandlerClass.model = model
        self.RequestHandlerClass.verbose = verbose
        HTTPServer.serve_forever(self)

class OnnxModel:
    def __init__(self, data, file):
        self.data = data
        self.file = file
    def optimize(self):
        # Remove raw initializer data
        model = onnx_ml_pb2.ModelProto()
        model.ParseFromString(self.data)
        for initializer in model.graph.initializer:
            self.remove_tensor_data(initializer)
        for node in model.graph.node:
            for attribute in node.attribute:
                if attribute.t:
                    self.remove_tensor_data(attribute.t)
        self.data = model.SerializeToString()
    def remove_tensor_data(self, tensor):
        del tensor.string_data[:]
        del tensor.int32_data[:]
        del tensor.int64_data[:]
        del tensor.float_data[:]
        tensor.raw_data = ""

class TensorFlowLiteModel:
    def __init__(self, data, file):
        self.data = data
        self.file = file
    def optimize(self):
        return

def serve_data(data, file, verbose=False, browse=False, port=8080, host='localhost', tensor=False):
    server = MyHTTPServer((host, port), MyHTTPRequestHandler)
    model = None
    if file.endswith('.tflite'):
        model = TensorFlowLiteModel(data, file)
    elif os.path.basename(file) == 'saved_model.pb':
        print('Not supported.')
        return
    else:
        model = OnnxModel(data, file)
    if not tensor:
        print("Processing '" + file + "'...")
        model.optimize()
    url = 'http://' + host + ':' + str(port)
    print("Serving '" + file + "' at " + url + "...")
    if browse:
        webbrowser.open(url);
    sys.stdout.flush()
    server.serve_forever(model, verbose)

def serve_file(file, verbose=False, browse=False, port=8080, host='localhost', tensor=False):
    print("Reading '" + file + "'...")
    data = None
    with open(file, 'rb') as binary:
        data = binary.read()
    serve_data(data, file, verbose=verbose, browse=browse, port=port, host=host, tensor=tensor)
