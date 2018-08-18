#!/usr/bin/python

import codecs
import os
import platform
import sys
import threading
import webbrowser

from .__version__ import __version__

if sys.version_info[0] > 2:
    from urllib.parse import urlparse
    from http.server import HTTPServer
    from http.server import BaseHTTPRequestHandler
    from socketserver import ThreadingMixIn
else:
    from urlparse import urlparse
    from BaseHTTPServer import HTTPServer
    from BaseHTTPServer import BaseHTTPRequestHandler
    from SocketServer import ThreadingMixIn

class HTTPRequestHandler(BaseHTTPRequestHandler):
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
        data = '/data/'
        if status_code == 0:
            if pathname == '/':
                meta = []
                meta.append("<meta name='type' content='Python' />")
                if __version__:
                    meta.append("<meta name='version' content='" + __version__ + "' />")
                if self.file:
                    meta.append("<meta name='file' content='/data/" + self.file + "' />")
                with codecs.open(location + 'view-browser.html', mode="r", encoding="utf-8") as open_file:
                    buffer = open_file.read()
                buffer = buffer.replace('<!-- meta -->', '\n'.join(meta))
                buffer = buffer.encode('utf-8');
                headers['Content-Type'] = 'text/html'
                headers['Content-Length'] = len(buffer)
                status_code = 200
            elif pathname.startswith(data):
                file = pathname[len(data):]
                if file == self.file:
                    buffer = self.data
                else:
                    file = self.folder + '/' + file;
                    status_code = 404
                    if os.path.exists(file):
                        with open(file, 'rb') as binary:
                            buffer = binary.read()
                if buffer:
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
                self.wfile.write(bytes(status_code))
            elif (status_code == 200 or status_code == 404) and buffer != None:
                self.wfile.write(buffer)
        return
    def do_GET(self):
        self.handler()
    def do_HEAD(self):
        self.handler()
    def log_message(self, format, *args):
        return

class ThreadedHTTPServer(ThreadingMixIn, HTTPServer): pass

def serve_data(data, file, verbose=False, browse=False, port=8080, host='localhost'):
    server = ThreadedHTTPServer((host, port), HTTPRequestHandler)
    server.RequestHandlerClass.folder = os.path.dirname(file) if file else ''
    server.RequestHandlerClass.file = os.path.basename(file) if file else ''
    server.RequestHandlerClass.data = data
    server.RequestHandlerClass.verbose = verbose
    url = 'http://' + host + ':' + str(port)
    if file:
        print("Serving '" + file + "' at " + url)
    else:
        print("Serving at " + url)
    sys.stdout.flush()
    if browse:
        threading.Timer(1, webbrowser.open, args=(url,)).start()
    try:
        while True:
            sys.stdout.flush()
            server.handle_request()
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

def browse(file, verbose=False, port=8080, host='localhost'):
    serve_file(file, verbose, True, port, host)
