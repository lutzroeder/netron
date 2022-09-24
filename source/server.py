''' Python Server implementation '''

import errno
import http.server
import importlib.util
import json
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

__version__ = '0.0.0'

class _ContentProvider: # pylint: disable=too-few-public-methods
    data = bytearray()
    base_dir = ''
    base = ''
    title = ''
    def __init__(self, data, path, file):
        self.data = data if data else bytearray()
        self.title = os.path.basename(file) if file else ''
        if path:
            self.dir = os.path.dirname(path) if os.path.dirname(path) else '.'
            self.base = os.path.basename(path)
    def read(self, path):
        ''' Read content '''
        if path == self.base and self.data:
            return self.data
        base_dir = os.path.realpath(self.dir)
        filename = os.path.normpath(os.path.realpath(base_dir + '/' + path))
        if os.path.commonprefix([ base_dir, filename ]) == base_dir:
            if os.path.exists(filename) and not os.path.isdir(filename):
                with open(filename, 'rb') as file:
                    return file.read()
        return None

class _HTTPRequestHandler(http.server.BaseHTTPRequestHandler):
    content = None
    verbosity = 1
    mime_types = {
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
    def do_HEAD(self): # pylint: disable=invalid-name
        ''' Serve a HEAD request '''
        self.do_GET()
    def do_GET(self): # pylint: disable=invalid-name
        ''' Serve a GET request '''
        path = urllib.parse.urlparse(self.path).path
        path = '/index.html' if path == '/' else path
        status_code = 404
        content = None
        content_type = None
        if path.startswith('/data/'):
            path = urllib.parse.unquote(path[len('/data/'):])
            content = self.content.read(path)
            if content:
                content_type = 'application/octet-stream'
                status_code = 200
        else:
            base_dir = os.path.dirname(os.path.realpath(__file__))
            filename = os.path.normpath(os.path.realpath(base_dir + path))
            extension = os.path.splitext(filename)[1]
            if os.path.commonprefix([base_dir, filename]) == base_dir and \
                os.path.exists(filename) and not os.path.isdir(filename) and \
                extension in self.mime_types:
                content_type = self.mime_types[extension]
                with open(filename, 'rb') as file:
                    content = file.read()
                if path == '/index.html':
                    content = content.decode('utf-8')
                    meta = [
                        '<meta name="type" content="Python">',
                        '<meta name="version" content="' + __version__ + '">'
                    ]
                    if self.content.base:
                        meta.append('<meta name="file" content="/data/' + self.content.base + '">')
                        content = re.sub(r'<title>.*</title>', \
                            '<title>' + self.content.title + '</title>', content)
                    meta = '\n'.join(meta)
                    content = re.sub(r'<meta name="version" content=".*">', meta, content)
                    content = content.encode('utf-8')
                status_code = 200
        _log(self.verbosity > 1, str(status_code) + ' ' + self.command + ' ' + self.path + '\n')
        self._write(status_code, content_type, content)
    def log_message(self, format, *args): # pylint: disable=redefined-builtin
        return
    def _write(self, status_code, content_type, content):
        self.send_response(status_code)
        if content:
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', len(content))
        self.end_headers()
        if self.command != 'HEAD':
            if status_code == 404 and content is None:
                self.wfile.write(str(status_code).encode('utf-8'))
            elif (status_code in (200, 404)) and content is not None:
                self.wfile.write(content)

class _ThreadedHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    pass

class _HTTPServerThread(threading.Thread):
    def __init__(self, content, address, verbosity):
        threading.Thread.__init__(self)
        self.verbosity = verbosity
        self.address = address
        self.url = 'http://' + address[0] + ':' + str(address[1])
        self.server = _ThreadedHTTPServer(address, _HTTPRequestHandler)
        self.server.timeout = 0.25
        self.server.RequestHandlerClass.content = content
        self.server.RequestHandlerClass.verbosity = verbosity
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
        ''' Stop server '''
        if self.alive():
            _log(self.verbosity > 0, "Stopping " + self.url + "\n")
            self.stop_event.set()
            self.server.server_close()
            self.terminate_event.wait(1000)

    def alive(self):
        ''' Check server status '''
        value = not self.terminate_event.is_set()
        return value

def _open(data):
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
                module = importlib.import_module(module_name, package=__package__)
                model_factory = module.ModelFactory()
                return model_factory.open(data)
        queue.extend(_ for _ in current.__bases__ if isinstance(_, type))
    return None

def _threads(address=None):
    threads = [ _ for _ in threading.enumerate() if isinstance(_, _HTTPServerThread) and _.alive() ]
    if address is not None:
        address = _make_address(address)
        threads = [ _ for _ in threads if address[0] == _.address[0] ]
        if address[1]:
            threads = [ _ for _ in threads if address[1] == _.address[1] ]
    return threads

def _log(condition, message):
    if condition:
        sys.stdout.write(message)
        sys.stdout.flush()

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
    threads = _threads(address)
    for thread in threads:
        thread.stop()

def status(adrress=None):
    '''Is model served at address.

    Args:
        address (tuple, optional): A (host, port) tuple, or a port number.
    '''
    threads = _threads(adrress)
    return len(threads) > 0

def wait():
    '''Wait for console exit and stop all model servers.'''
    try:
        while len(_threads()) > 0:
            time.sleep(0.1)
    except (KeyboardInterrupt, SystemExit):
        _log(True, '\n')
        stop()

def serve(file, data, address=None, browse=False, verbosity=1):
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
    verbosity = { '0': 0, 'quiet': 0, '1': 1, 'default': 1, '2': 2, 'debug': 2 }[str(verbosity)]

    if not data and file and not os.path.exists(file):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), file)

    content = _ContentProvider(data, file, file)

    if data and not isinstance(data, bytearray) and isinstance(data.__class__, type):
        _log(verbosity > 1, 'Experimental\n')
        model = _open(data)
        if model:
            text = json.dumps(model.to_json(), indent=4, ensure_ascii=False)
            content = _ContentProvider(text.encode('utf-8'), 'model.netron', file)

    address = _make_address(address)
    if isinstance(address[1], int) and address[1] != 0:
        stop(address)
    else:
        address = _make_port(address)

    thread = _HTTPServerThread(content, address, verbosity)
    thread.start()
    while not thread.alive():
        time.sleep(0.01)
    message = (("Serving '" + file + "'") if file else "Serving") + " at " + thread.url + "\n"
    _log(verbosity > 0, message)
    if browse:
        webbrowser.open(thread.url)

    return address

def start(file=None, address=None, browse=True, verbosity=1):
    '''Start serving model file at address and open in web browser.

    Args:
        file (string): Model file to serve.
        log (bool, optional): Log details to console. Default: False
        browse (bool, optional): Launch web browser, Default: True
        address (tuple, optional): A (host, port) tuple, or a port number.

    Returns:
        A (host, port) address tuple.
    '''
    return serve(file, None, browse=browse, address=address, verbosity=verbosity)
