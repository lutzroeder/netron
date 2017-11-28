#!/usr/bin/python

import os
import platform
import sys
import webbrowser
import onnx

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
        if pathname == '/':
            pathname = '/view-browser.html'
        location = folder + pathname;
        status_code = 0
        headers = {}
        buffer = None
        if status_code == 0:
            if os.path.exists(location) and os.path.isdir(location):
                if location.endswith('/'):
                    location += 'view-browser.html'
                else:
                    status_code = 302
                    headers = { 'Location': pathname + '/' }
        if status_code == 0:
            if pathname == '/model':
                buffer = self.data
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
    def serve_forever(self, data):
        self.RequestHandlerClass.data = data 
        HTTPServer.serve_forever(self)

def serve_data(data, browse=False, port=8080, initializer=False):
    server = MyHTTPServer(('localhost', port), MyHTTPRequestHandler)
    if not initializer:
        # Remove raw initializer data
        model = onnx.ModelProto()
        model.ParseFromString(data)
        for initializer in model.graph.initializer:
          initializer.raw_data = ""
        data = model.SerializeToString()
    if browse:
        webbrowser.open('http://localhost:' + str(port));
    sys.stdout.flush()
    server.serve_forever(data)

def serve_file(file, browse=False, port=8080, initializer=False):
    data = None
    with open(file, 'rb') as binary:
        data = binary.read()
    url = 'http://localhost:' + str(port)
    print("Serving '" + file + "' at " + url + "...")
    serve_data(data, browse, port, initializer)
