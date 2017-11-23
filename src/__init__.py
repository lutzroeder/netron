#!/usr/bin/python

import codecs
import os
import platform
import re
import sys
import base64

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
        pathname = urlparse(self.path).path
        folder = os.path.dirname(os.path.realpath(__file__))
        if pathname == '/':
            pathname = '/view-browser.html'
        location = folder + pathname;
        if pathname == '/model':
            location = self.model_file;
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
            if os.path.exists(location) and not os.path.isdir(location):
                status_code = 200
            else:
                status_code = 404
            if os.path.exists(location) and not os.path.isdir(location):
                with open(location, 'rb') as binary:
                    buffer = binary.read()
                headers['Content-Length'] = len(buffer)
                extension = os.path.splitext(location)[1]
                if pathname == '/model':
                    content_type = 'text/plain'
                    buffer = base64.b64encode(buffer)
                    headers['Content-Length'] = len(buffer)
                else:
                    if not hasattr(self, 'mime_types_map'):
                        self.mime_types_map = {
                            '.html': 'text/html',
                            '.js':   'text/javascript',
                            '.css':  'text/css',
                            '.png':  'image/png',
                            '.gif':  'image/gif',
                            '.jpg':  'image/jpeg',
                            '.json': 'application/json',
                            '.ttf': 'font/truetype',
                            '.woff': 'font/woff',
                            '.otf': 'font/opentype',
                            '.eot': 'application/vnd.ms-fontobject',
                            '.woff': 'application/font-woff',
                            '.woff2': 'application/font-woff2',
                            '.svg': 'image/svg+xml'
                        }
                    content_type = self.mime_types_map[extension]
                if content_type:
                    headers['Content-Type'] = content_type
        # print(str(status_code) + ' ' + self.command + ' ' + self.path)
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
    def serve_forever(self, model_file):
        self.RequestHandlerClass.model_file = model_file 
        HTTPServer.serve_forever(self)

def show_help():
    print('')
    print('Usage:')
    print('  netron [option(s)] <model-file>')
    print('')
    print('Options:')
    print('  --help          Show help.')
    print('  --port <port>   Port to serve (default: 8080).')
    print('  --browse        Launch web browser.')
    print('')

def serve(args):
    port = 8080
    browse = False
    model_file = ''
    while len(args) > 0:
        arg = args.pop(0)
        if (arg == '--help' or arg == '-h'):
            show_help()
            return
        elif (arg == '--port' or arg == '-p') and len(args) > 0 and args[0].isdigit(): 
            port = int(args.pop(0))
        elif arg == '--browse' or arg == '-b':
            browse = True
        elif not arg.startswith('-'):
            model_file = arg
    if len(model_file) == 0:
        show_help()
        return
    if not os.path.exists(model_file):
        print("Model file '" + model_file + "' does not exist.")
        return
    server = MyHTTPServer(('localhost', port), MyHTTPRequestHandler)
    url = 'http://localhost:' + str(port)
    print("Serving '" + model_file + "' at " + url + "...")
    if browse:
        command = 'xdg-open';
        if platform.system() == 'Darwin':
            command = 'open'
        elif platform.system() == 'Windows':
            command = 'start ""'
        os.system(command + ' "' + url.replace('"', '\"') + '"')
    sys.stdout.flush()
    server.serve_forever(model_file)
