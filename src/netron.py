#!/usr/bin/python

import codecs
import os
import platform
import sys
import threading
import webbrowser

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
                if self.file:
                    buffer = buffer.replace('<!-- meta -->', '<meta name=\'file\' content=\'' + self.file + '\'>')
                buffer = buffer.encode('utf-8');
                headers['Content-Type'] = 'text/html'
                headers['Content-Length'] = len(buffer)
                status_code = 200
            elif pathname == '/data' and self.data:
                buffer = self.data
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
    def initialize_data(self, data, file, verbose):
        self.RequestHandlerClass.file = file
        self.RequestHandlerClass.data = data
        self.RequestHandlerClass.verbose = verbose

def serve_data(data, file, verbose=False, browse=False, port=8080, host='localhost'):
    server = MyHTTPServer((host, port), MyHTTPRequestHandler)
    url = 'http://' + host + ':' + str(port)
    if file:
        print("Serving '" + file + "' at " + url)
    else:
        print("Serving at " + url)
    server.initialize_data(data, file, verbose)
    sys.stdout.flush()
    if browse:
        threading.Timer(1, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    except (KeyboardInterrupt, SystemExit):
        print("\nStopping")
        server.server_close()

def serve_file(file, verbose=False, browse=False, port=8080, host='localhost'):
    data = None
    if file and os.path.exists(file):
        print("Reading '" + file + "'")
        with open(file, 'rb') as binary:
            data = binary.read()
    else:
        file = None
    serve_data(data, file, verbose=verbose, browse=browse, port=port, host=host)
