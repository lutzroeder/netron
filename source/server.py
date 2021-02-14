
import codecs
import errno
import os
import random
import re
import socket
import sys
import threading
import webbrowser
import time

from .__version__ import __version__

if sys.version_info[0] > 2:
    from urllib.parse import urlparse
    from urllib.parse import unquote
    from http.server import HTTPServer
    from http.server import BaseHTTPRequestHandler
    from socketserver import ThreadingMixIn
else:
    from urlparse import urlparse
    from urlparse import unquote
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
                '.otf': 'font/opentype',
                '.eot': 'application/vnd.ms-fontobject',
                '.woff': 'font/woff',
                '.woff2': 'application/font-woff2',
                '.svg': 'image/svg+xml'
            }
        pathname = urlparse(self.path).path
        folder = os.path.dirname(os.path.realpath(__file__))
        location = folder + pathname
        status_code = 0
        headers = {}
        buffer = None
        data = '/data/'
        if status_code == 0:
            if pathname == '/':
                meta = []
                meta.append('<meta name="type" content="Python">')
                meta.append('<meta name="version" content="' + __version__ + '">')
                if self.file:
                    meta.append('<meta name="file" content="/data/' + self.file + '">')
                with codecs.open(location + 'index.html', mode="r", encoding="utf-8") as open_file:
                    buffer = open_file.read()
                buffer = re.sub(r'<meta name="version" content="\d+.\d+.\d+">', '\n'.join(meta), buffer)
                buffer = buffer.encode('utf-8')
                headers['Content-Type'] = 'text/html'
                headers['Content-Length'] = len(buffer)
                status_code = 200
            elif pathname.startswith(data):
                file = pathname[len(data):]
                if file == self.file and self.data:
                    buffer = self.data
                else:
                    file = self.folder + '/' + unquote(file)
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
        if self.log:
            sys.stdout.write(str(status_code) + ' ' + self.command + ' ' + self.path + '\n')
        sys.stdout.flush()
        self.send_response(status_code)
        for key in headers:
            self.send_header(key, headers[key])
        self.end_headers()
        if self.command != 'HEAD':
            if status_code == 404 and buffer is None:
                self.wfile.write(bytes(status_code))
            elif (status_code in (200, 404)) and buffer is not None:
                self.wfile.write(buffer)
    def do_GET(self):
        self.handler()
    def do_HEAD(self):
        self.handler()
    def log_message(self, format, *args):
        return

class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    pass

class HTTPServerThread(threading.Thread):
    def __init__(self, data, file, address, log):
        threading.Thread.__init__(self)
        self.address = address
        self.url = 'http://' + address[0] + ':' + str(address[1])
        self.file = file
        self.server = ThreadedHTTPServer(address, HTTPRequestHandler)
        self.server.timeout = 0.25
        if file:
            self.server.RequestHandlerClass.folder = os.path.dirname(file) if os.path.dirname(file) else '.'
            self.server.RequestHandlerClass.file = os.path.basename(file)
        else:
            self.server.RequestHandlerClass.folder = ''
            self.server.RequestHandlerClass.file = ''
        self.server.RequestHandlerClass.data = data
        self.server.RequestHandlerClass.log = log
        self.terminate_event = threading.Event()
        self.terminate_event.set()
        self.stop_event = threading.Event()

    def run(self):
        self.stop_event.clear()
        self.terminate_event.clear()
        try:
            while not self.stop_event.is_set():
                self.server.handle_request()
        except Exception:
            pass
        self.terminate_event.set()
        self.stop_event.clear()

    def stop(self):
        if self.alive():
            sys.stdout.write("Stopping " + self.url + "\n")
            self.stop_event.set()
            self.server.server_close()
            self.terminate_event.wait(1000)

    def alive(self):
        return not self.terminate_event.is_set()

_thread_list = []

def _add_thread(thread):
    global _thread_list
    _thread_list.append(thread)

def _update_thread_list():
    global _thread_list
    _thread_list = [ thread for thread in _thread_list if thread.alive() ]
    return _thread_list

def _make_address(address):
    if address is None or isinstance(address, int):
        port = address
        address = ('localhost', port)
    if isinstance(address, tuple) and len(address) == 2:
        host = address[0]
        port = address[1]
        if isinstance(host, str) and (port is None or isinstance(port, int)):
            return address
    raise ValueError('Invalid address.')

def _make_port(address):
    if address[1] is None or address[1] == 0:
        ports = []
        if address[1] != 0:
            ports.append(8080)
            ports.append(8081)
            rnd = random.Random()
            for _ in range(4):
                port = rnd.randrange(15000, 25000)
                if port not in ports:
                    ports.append(port)
        ports.append(0)
        for port in ports:
            temp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            temp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            temp_socket.settimeout(1)
            try:
                temp_socket.bind((address[0], port))
                sockname = temp_socket.getsockname()
                address = (address[0], sockname[1])
                return address
            except:
                pass
            finally:
                temp_socket.close()
    if isinstance(address[1], int):
        return address
    raise ValueError('Failed to allocate port.')

def stop(address=None):
    '''Stop serving model at address.

    Args:
        address (tuple, optional): A (host, port) tuple, or a port number.
    '''
    threads = _update_thread_list()
    if address is not None:
        address = _make_address(address)
        if address[1] is None:
            threads = [ thread for thread in threads if address[0] == thread.address[0] ]
        else:
            threads = [ thread for thread in threads if address[0] == thread.address[0] and address[1] == thread.address[1] ]
    for thread in threads:
        thread.stop()
    _update_thread_list()

def wait():
    '''Wait for console exit and stop all model servers.'''
    try:
        while len(_update_thread_list()) > 0:
            time.sleep(1000)
    except (KeyboardInterrupt, SystemExit):
        sys.stdout.write('\n')
        sys.stdout.flush()
        stop()

def serve(file, data, address=None, browse=False, log=False):
    '''Start serving model from file or data buffer at address and open in web browser.

    Args:
        file (string): Model file to serve. Required to detect format.
        data (bytes): Model data to serve. None will load data from file.
        log (bool, optional): Log details to console. Default: False
        browse (bool, optional): Launch web browser. Default: True
        address (tuple, optional): A (host, port) tuple, or a port number.

    Returns:
        A (host, port) address tuple.
    '''
    if not data and file and not os.path.exists(file):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), file)

    _update_thread_list()
    address = _make_address(address)
    if isinstance(address[1], int) and address[1] != 0:
        stop(address)
    else:
        address = _make_port(address)
    _update_thread_list()

    thread = HTTPServerThread(data, file, address, log)
    thread.start()
    while not thread.alive():
        time.sleep(10)
    _add_thread(thread)

    if file:
        sys.stdout.write("Serving '" + file + "' at " + thread.url + "\n")
    else:
        sys.stdout.write("Serving at " + thread.url + "\n")
    sys.stdout.flush()
    if browse:
        webbrowser.open(thread.url)

    return address

def start(file=None, address=None, browse=True, log=False):
    '''Start serving model file at address and open in web browser.

    Args:
        file (string): Model file to serve.
        log (bool, optional): Log details to console. Default: False
        browse (bool, optional): Launch web browser, Default: True
        address (tuple, optional): A (host, port) tuple, or a port number.

    Returns:
        A (host, port) address tuple.
    '''
    return serve(file, None, browse=browse, address=address, log=log)
