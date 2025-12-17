""" Python Server implementation """

import base64
import errno
import http.server
import importlib
import importlib.metadata
import json
import logging
import os
import random
import re
import socket
import socketserver
import tempfile
import threading
import time
import traceback
import urllib.parse
import webbrowser

__version__ = "0.0.0"

logger = logging.getLogger(__name__)

# OnnxSlim integration
_onnxslim_bridge = None

def _get_onnxslim_bridge():
    """Lazy load OnnxSlim bridge module."""
    global _onnxslim_bridge
    if _onnxslim_bridge is None:
        try:
            from . import onnxslim_bridge
            _onnxslim_bridge = onnxslim_bridge
        except ImportError:
            # Try direct import if not in package
            try:
                import onnxslim_bridge
                _onnxslim_bridge = onnxslim_bridge
            except ImportError:
                _onnxslim_bridge = False
    return _onnxslim_bridge if _onnxslim_bridge else None

class _ContentProvider:
    def __init__(self, data, path, file, name):
        self.data = data if data else bytearray()
        self.identifier = os.path.basename(file) if file else ""
        self.name = name
        self.dir = ""
        self.base = ""
        self.full_path = ""
        if path:
            self.dir = os.path.dirname(path) if os.path.dirname(path) else "."
            self.base = os.path.basename(path)
            # Store full absolute path for OnnxSlim integration
            self.full_path = os.path.realpath(path)
    def read(self, path):
        if path == self.base and self.data:
            return self.data
        base_dir = os.path.realpath(self.dir)
        filename = os.path.normpath(os.path.realpath(base_dir + "/" + path))
        if os.path.commonpath([ base_dir, filename ]) == base_dir:
            if os.path.exists(filename) and not os.path.isdir(filename):
                with open(filename, "rb") as file:
                    return file.read()
        return None

class _HTTPRequestHandler(http.server.BaseHTTPRequestHandler):
    content = None
    mime_types = {
        ".html": "text/html",
        ".js":   "text/javascript",
        ".css":  "text/css",
        ".png":  "image/png",
        ".gif":  "image/gif",
        ".jpg":  "image/jpeg",
        ".ico":  "image/x-icon",
        ".json": "application/json",
        ".pb": "application/octet-stream",
        ".ttf": "font/truetype",
        ".otf": "font/opentype",
        ".eot": "application/vnd.ms-fontobject",
        ".woff": "font/woff",
        ".woff2": "font/woff2",
        ".svg": "image/svg+xml",
        ".onnx": "application/octet-stream"
    }
    def do_HEAD(self):
        self.do_GET()
    def do_OPTIONS(self):
        """Handle CORS preflight requests."""
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
    def do_POST(self):
        """Handle POST requests for OnnxSlim API."""
        path = urllib.parse.urlparse(self.path).path
        if path.startswith("/api/onnxslim"):
            self._handle_onnxslim_api()
        else:
            self._write(404, "application/json", json.dumps({"error": "Not found"}).encode("utf-8"))
    def _handle_onnxslim_api(self):
        """Handle OnnxSlim API requests."""
        bridge = _get_onnxslim_bridge()
        if not bridge:
            response = {
                "status": "error",
                "error": "OnnxSlim bridge not available. Make sure OnnxSlim is installed."
            }
            self._write(500, "application/json", json.dumps(response).encode("utf-8"))
            return
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            request_data = json.loads(body.decode("utf-8")) if body else {}

            # Check if model data is sent as base64 from browser
            model_data_base64 = request_data.get("model_data_base64")
            if model_data_base64:
                # Decode base64 to bytes and use for in-memory optimization
                try:
                    model_data = base64.b64decode(model_data_base64)
                    request_data["model_data"] = model_data
                    request_data["model_name"] = request_data.get("model_name", "model.onnx")
                    request_data["input_path"] = None  # Clear path since we're using data
                    # Remove base64 data from request to save memory
                    del request_data["model_data_base64"]
                    logger.info(f"OnnxSlim using base64 in-memory model: {request_data.get('model_name')}")
                except Exception as e:
                    response = {
                        "status": "error",
                        "error": f"Failed to decode base64 model data: {str(e)}"
                    }
                    self._write(400, "application/json", json.dumps(response).encode("utf-8"))
                    return
            else:
                # Resolve input path or get model data from content provider
                input_path = request_data.get("input_path", "")
                model_name = ""

                # If no input_path provided, try to use the current model's full path
                if not input_path:
                    if hasattr(self.content, "full_path") and self.content.full_path:
                        input_path = self.content.full_path
                    elif hasattr(self.content, "base") and self.content.base:
                        input_path = self.content.base
                    model_name = getattr(self.content, "identifier", "") or getattr(self.content, "base", "") or "model"

                if input_path:
                    resolved_path = None

                    # If input_path is absolute and exists, use it directly
                    if os.path.isabs(input_path) and os.path.exists(input_path):
                        resolved_path = input_path
                    else:
                        # Try to use the stored full_path if the basename matches
                        if hasattr(self.content, "full_path") and self.content.full_path:
                            if os.path.basename(self.content.full_path) == input_path or input_path == self.content.base:
                                if os.path.exists(self.content.full_path):
                                    resolved_path = self.content.full_path

                        # Try to construct full path from content directory
                        if not resolved_path and hasattr(self.content, "dir") and self.content.dir:
                            full_path = os.path.join(self.content.dir, input_path)
                            full_path = os.path.realpath(full_path)
                            if os.path.exists(full_path):
                                resolved_path = full_path

                    # If path still doesn't exist, try to get data from content provider
                    if not resolved_path:
                        model_data = None
                        # Try reading via content provider
                        if hasattr(self.content, "read"):
                            model_data = self.content.read(input_path)
                        # Also check if data is stored directly in content
                        if not model_data and hasattr(self.content, "data") and self.content.data:
                            model_data = bytes(self.content.data)

                        if model_data:
                            # Pass model data directly to bridge (in-memory optimization)
                            request_data["model_data"] = model_data
                            request_data["model_name"] = model_name or input_path
                            request_data["input_path"] = None  # Clear path since we're using data
                            logger.info(f"OnnxSlim using in-memory model: {model_name or input_path}")
                        else:
                            response = {
                                "status": "error",
                                "error": f"Input file not found: {input_path}. Could not resolve path or read model data."
                            }
                            self._write(400, "application/json", json.dumps(response).encode("utf-8"))
                            return
                    else:
                        request_data["input_path"] = resolved_path
                        logger.info(f"OnnxSlim using input path: {resolved_path}")

            result = bridge.handle_request(request_data)
            response_body = json.dumps(result).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", len(response_body))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(response_body)
        except json.JSONDecodeError as e:
            response = {"status": "error", "error": f"Invalid JSON: {str(e)}"}
            self._write(400, "application/json", json.dumps(response).encode("utf-8"))
        except Exception as e:
            logger.error(f"OnnxSlim API error: {e}")
            logger.error(traceback.format_exc())
            response = {"status": "error", "error": str(e)}
            self._write(500, "application/json", json.dumps(response).encode("utf-8"))
    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        query = urllib.parse.parse_qs(parsed.query)
        path = "/index.html" if path == "/" else path
        status_code = 404
        content = None
        content_type = None

        # Handle download of optimized models from temp directory
        if path == "/api/download":
            file_path = query.get("path", [None])[0]
            if file_path:
                file_path = urllib.parse.unquote(file_path)
                # Only allow downloading from temp directory for security
                temp_dir = tempfile.gettempdir()
                real_path = os.path.realpath(file_path)
                if real_path.startswith(temp_dir) and os.path.exists(real_path) and os.path.isfile(real_path):
                    with open(real_path, "rb") as f:
                        content = f.read()
                    content_type = "application/octet-stream"
                    status_code = 200
                    # Add Content-Disposition header for download
                    self.send_response(status_code)
                    self.send_header("Content-Type", content_type)
                    self.send_header("Content-Length", len(content))
                    self.send_header("Content-Disposition", f'attachment; filename="{os.path.basename(real_path)}"')
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    self.wfile.write(content)
                    return
        # Handle switching to optimized model
        elif path == "/api/open":
            file_path = query.get("path", [None])[0]
            if file_path:
                file_path = urllib.parse.unquote(file_path)
                # Only allow opening from temp directory for security
                temp_dir = tempfile.gettempdir()
                real_path = os.path.realpath(file_path)
                if real_path.startswith(temp_dir) and os.path.exists(real_path) and os.path.isfile(real_path):
                    # Update content provider to serve the new file
                    self.content.dir = os.path.dirname(real_path)
                    self.content.base = os.path.basename(real_path)
                    self.content.full_path = real_path
                    self.content.identifier = os.path.basename(real_path)
                    self.content.name = real_path
                    self.content.data = bytearray()  # Clear any cached data
                    # Redirect to home page with cache buster
                    self.send_response(302)
                    self.send_header("Location", "/?t=" + str(int(time.time() * 1000)))
                    self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
                    self.end_headers()
                    return
        elif path.startswith("/data/"):
            path = urllib.parse.unquote(path[len("/data/"):])
            content = self.content.read(path)
            if content:
                content_type = "application/octet-stream"
                status_code = 200
        else:
            base_dir = os.path.dirname(os.path.realpath(__file__))
            filename = os.path.normpath(os.path.realpath(base_dir + path))
            extension = os.path.splitext(filename)[1]
            if os.path.commonpath([base_dir, filename]) == base_dir and \
                os.path.exists(filename) and not os.path.isdir(filename) and \
                extension in self.mime_types:
                content_type = self.mime_types[extension]
                with open(filename, "rb") as file:
                    content = file.read()
                if path == "/index.html":
                    content = content.decode("utf-8")
                    meta = [
                        '<meta name="type" content="Python">',
                        '<meta name="version" content="' + __version__ + '">'
                    ]
                    base = self.content.base
                    if base:
                        meta.append('<meta name="file" content="/data/' + base + '">')
                    name = self.content.name
                    if name:
                        meta.append('<meta name="name" content="' + name + '">')
                    identifier = self.content.identifier
                    if identifier:
                        meta.append(f'<meta name="identifier" content="{identifier}">')
                    meta = "\n".join(meta)
                    regex = r'<meta name="version" content=".*">'
                    content = re.sub(regex, lambda _: meta, content)
                    content = content.encode("utf-8")
                    # Serve index.html with no-cache headers
                    self.send_response(200)
                    self.send_header("Content-Type", content_type)
                    self.send_header("Content-Length", len(content))
                    self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
                    self.end_headers()
                    self.wfile.write(content)
                    return
                status_code = 200
        self._write(status_code, content_type, content)
    def log_message(self, format, *args):
        logger.debug(" ".join(args))
    def _write(self, status_code, content_type, content):
        self.send_response(status_code)
        if content:
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", len(content))
        self.end_headers()
        if self.command != "HEAD":
            if status_code == 404 and content is None:
                self.wfile.write(str(status_code).encode("utf-8"))
            elif (status_code in (200, 404)) and content is not None:
                self.wfile.write(content)

class _ThreadedHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    pass

class _HTTPServerThread(threading.Thread):
    def __init__(self, content, address):
        threading.Thread.__init__(self)
        self.daemon = True
        self.address = address
        self.url = "http://" + address[0] + ":" + str(address[1])
        self.server = _ThreadedHTTPServer(address, _HTTPRequestHandler)
        self.server.timeout = 0.25
        self.server.block_on_close = False
        self.server.RequestHandlerClass.content = content
        self.terminate_event = threading.Event()
        self.terminate_event.set()
        self.stop_event = threading.Event()

    def run(self):
        self.stop_event.clear()
        self.terminate_event.clear()
        try:
            while not self.stop_event.is_set():
                self.server.handle_request()
        except: # noqa: E722
            pass
        self.terminate_event.set()
        self.stop_event.clear()

    def stop(self):
        if self.alive():
            logger.info("Stopping " + self.url)
            self.stop_event.set()
            self.server.server_close()

    def alive(self):
        value = not self.terminate_event.is_set()
        return value

def _open(data):
    registry = dict([
        ("onnx.onnx_ml_pb2.ModelProto", ".onnx"),
        ("torch.jit._script.ScriptModule", ".pytorch"),
        ("torch.Graph", ".pytorch"),
        ("torch._C.Graph", ".pytorch"),
        ("torch.nn.modules.module.Module", ".pytorch")
    ])
    queue = [ data.__class__ ]
    while len(queue) > 0:
        current = queue.pop(0)
        if current.__module__ and current.__name__:
            name = current.__module__ + "." + current.__name__
            if name in registry:
                module_name = registry[name]
                module = importlib.import_module(module_name, package=__package__)
                model_factory = module.ModelFactory()
                return model_factory.open(data)
        queue.extend(_ for _ in current.__bases__ if isinstance(_, type))
    return None

def _threads(address=None):
    threads = []
    for thread in threading.enumerate():
        if isinstance(thread, _HTTPServerThread) and thread.alive():
            threads.append(thread)
    if address is not None:
        address = _make_address(address)
        threads = [ _ for _ in threads if address[0] == _.address[0] ]
        if address[1]:
            threads = [ _ for _ in threads if address[1] == _.address[1] ]
    return threads

def _make_address(address):
    if address is None or isinstance(address, int):
        port = address
        address = ("localhost", port)
    if isinstance(address, tuple) and len(address) == 2:
        host = address[0]
        port = address[1]
        if isinstance(host, str) and (port is None or isinstance(port, int)):
            return address
    raise ValueError("Invalid address.")

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
            except: # noqa: E722
                pass
            finally:
                temp_socket.close()
    if isinstance(address[1], int):
        return address
    raise ValueError("Failed to allocate port.")

def stop(address=None):
    """Stop serving model at address.

    Args:
        address (tuple, optional): A (host, port) tuple, or a port number.
    """
    threads = _threads(address)
    for thread in threads:
        thread.stop()

def status(address=None):
    """Is model served at address.

    Args:
        address (tuple, optional): A (host, port) tuple, or a port number.
    """
    threads = _threads(address)
    return len(threads) > 0

def wait():
    """Wait for console exit and stop all model servers."""
    try:
        while len(_threads()) > 0:
            time.sleep(0.1)
    except (KeyboardInterrupt, SystemExit):
        stop()

def serve(file, data=None, address=None, browse=False):
    """Start serving model from file or data buffer at address and open in web browser.

    Args:
        file (string): Model file to serve. Required to detect format.
        data (bytes): Model data to serve. None will load data from file.
        address (tuple, optional): A (host, port) tuple, or a port number.
        browse (bool, optional): Launch web browser. Default: True

    Returns:
        A (host, port) address tuple.
    """
    if not logging.getLogger().hasHandlers():
        logging.basicConfig(level=logging.INFO, format="%(message)s")

    if not data and file and not os.path.exists(file):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), file)

    content = _ContentProvider(data, file, file, file)

    if data and not isinstance(data, bytearray) and isinstance(data.__class__, type):
        logger.info("Experimental")
        model = _open(data)
        if model:
            text = json.dumps(model.to_json(), indent=2, ensure_ascii=False)
            content = _ContentProvider(text.encode("utf-8"), "model.netron", None, file)

    address = _make_address(address)
    if isinstance(address[1], int) and address[1] != 0:
        stop(address)
    else:
        address = _make_port(address)

    thread = _HTTPServerThread(content, address)
    thread.start()
    while not thread.alive():
        time.sleep(0.01)
    state = ("Serving '" + file + "'") if file else "Serving"
    logger.info(f"{state} at {thread.url}")
    if browse:
        webbrowser.open(thread.url)

    return address

def start(file=None, address=None, browse=True):
    """Start serving model file at address and open in web browser.

    Args:
        file (string): Model file to serve.
        browse (bool, optional): Launch web browser, Default: True
        address (tuple, optional): A (host, port) tuple, or a port number.

    Returns:
        A (host, port) address tuple.
    """
    return serve(file, None, browse=browse, address=address)

def widget(address, height=800):
    """ Open address as Jupyter Notebook IFrame.

    Args:
        address (tuple, optional): A (host, port) tuple, or a port number.
        height (int, optional): Height of the IFrame, Default: 800

    Returns:
        A Jupyter Notebook IFrame.
    """
    address = _make_address(address)
    url = f"http://{address[0]}:{address[1]}"
    IPython = __import__("IPython")
    return IPython.display.IFrame(url, width="100%", height=height)
