''' Python Server '''

import codecs
import errno
import http.server
import importlib.util
import os
import random
import re
import socket
import socketserver
import sys
import threading
import time
import webbrowser
import urllib.parse

from .__version__ import __version__

class HTTPRequestHandler(http.server.BaseHTTPRequestHandler):
    ''' HTTP Request Handler '''
    file = ""
    data = bytearray()
    folder = ""
    log = False
    mime_types_map = {
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
    def do_GET(self): # pylint: disable=invalid-name
        ''' Serve a GET request '''
        path = urllib.parse.urlparse(self.path).path
        status_code = 0
        headers = {}
        buffer = None
        if path == '/' or path == '/index.html':
            meta = []
            meta.append('<meta name="type" content="Python">')
            meta.append('<meta name="version" content="' + __version__ + '">')
            if self.file:
                meta.append('<meta name="file" content="/data/' + self.file + '">')
            basedir = os.path.dirname(os.path.realpath(__file__))
            with codecs.open(basedir + '/index.html', mode="r", encoding="utf-8") as open_file:
                buffer = open_file.read()
            meta = '\n'.join(meta)
            buffer = re.sub(r'<meta name="version" content="\d+.\d+.\d+">', meta, buffer)
            buffer = buffer.encode('utf-8')
            headers['Content-Type'] = 'text/html'
            headers['Content-Length'] = len(buffer)
            status_code = 200
        elif path.startswith('/data/'):
            status_code = 404
            path = urllib.parse.unquote(path[len('/data/'):])
            if path == self.file and self.data:
                buffer = self.data
            else:
                basedir = os.path.realpath(self.folder)
                path = os.path.normpath(os.path.realpath(basedir + '/' + path))
                if os.path.commonprefix([basedir, path]) == basedir:
                    if os.path.exists(path) and not os.path.isdir(path):
                        with open(path, 'rb') as file:
                            buffer = file.read()
            if buffer:
                headers['Content-Type'] = 'application/octet-stream'
                headers['Content-Length'] = len(buffer)
                status_code = 200
        else:
            status_code = 404
            basedir = os.path.dirname(os.path.realpath(__file__))
            path = os.path.normpath(os.path.realpath(basedir + path))
            if os.path.commonprefix([basedir, path]) == basedir:
                if os.path.exists(path) and not os.path.isdir(path):
                    extension = os.path.splitext(path)[1]
                    content_type = self.mime_types_map[extension]
                    if content_type:
                        with open(path, 'rb') as file:
                            buffer = file.read()
                        headers['Content-Type'] = content_type
                        headers['Content-Length'] = len(buffer)
                        status_code = 200
        if self.log:
            sys.stdout.write(str(status_code) + ' ' + self.command + ' ' + self.path + '\n')
        sys.stdout.flush()
        self.send_response(status_code)
        for key, value in headers.items():
            self.send_header(key, value)
        self.end_headers()
        if self.command != 'HEAD':
            if status_code == 404 and buffer is None:
                self.wfile.write(bytes(status_code))
            elif (status_code in (200, 404)) and buffer is not None:
                self.wfile.write(buffer)
    def do_HEAD(self): # pylint: disable=invalid-name
        ''' Serve a HEAD request '''
        self.do_GET()
    def log_message(self, format, *args): # pylint: disable=redefined-builtin
        return

class HTTPServerThread(threading.Thread):
    ''' HTTP Server Thread '''
    def __init__(self, data, file, address, log):
        threading.Thread.__init__(self)
        class ThreadedHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
            ''' Threaded HTTP Server '''
        self.address = address
        self.url = 'http://' + address[0] + ':' + str(address[1])
        self.file = file
        self.server = ThreadedHTTPServer(address, HTTPRequestHandler)
        self.server.timeout = 0.25
        if file:
            folder = os.path.dirname(file) if os.path.dirname(file) else '.'
            self.server.RequestHandlerClass.folder = folder
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
        except: # pylint: disable=bare-except
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

def _update_thread_list(address=None):
    global _thread_list
    _thread_list = [ thread for thread in _thread_list if thread.alive() ]
    threads = _thread_list
    if address is not None:
        address = _make_address(address)
        if address[1] is None:
            threads = [ _ for _ in threads if address[0] == _.address[0] ]
        else:
            threads = [ _ for _ in threads if address[0] == _.address[0] and address[1] == _.address[1] ]
    return threads

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
            except: # pylint: disable=bare-except
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
    threads = _update_thread_list(address)
    for thread in threads:
        thread.stop()
    _update_thread_list()

def status(adrress=None):
    '''Is model served at address.

    Args:
        address (tuple, optional): A (host, port) tuple, or a port number.
    '''
    threads = _update_thread_list(adrress)
    return len(threads) > 0

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
        address (tuple, optional): A (host, port) tuple, or a port number.
        browse (bool, optional): Launch web browser. Default: True
        log (bool, optional): Log details to console. Default: False

    Returns:
        A (host, port) address tuple.
    '''
    if not data and file and not os.path.exists(file):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), file)

    if data and not isinstance(data, bytearray) and isinstance(data.__class__, type):
        registry = dict([
            ('onnx.onnx_ml_pb2.ModelProto', '.onnx'),
            ('torch.Graph', '.pytorch'),
            ('torch._C.Graph', '.pytorch'),
            ('torch.nn.modules.module.Module', '.pytorch')
        ])
        queue = [ data.__class__ ]
        while len(queue) > 0:
            current = queue.pop(0)
            if current.__module__ and current.__name__:
                name = current.__module__ + '.' + current.__name__
                if name in registry:
                    module_name = registry[name]
                    if module_name.startswith('.'):
                        file = os.path.join(os.path.dirname(__file__), module_name[1:] + '.py')
                        spec = importlib.util.spec_from_file_location(module_name, file)
                        module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(module)
                    else:
                        module = __import__(module_name)
                    model_factory = module.ModelFactory()
                    data = model_factory.serialize(data)
                    file = 'test.json'
                    break
            for base in current.__bases__:
                if isinstance(base, type):
                    queue.append(base)
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
