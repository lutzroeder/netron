/*jshint esversion: 6 */

class FindView {

    constructor(graphElement, graph) {
        this._graphElement = graphElement;
        this._graph = graph;
        this._contentElement = document.createElement('div');
        this._contentElement.setAttribute('class', 'find');
        this._searchElement = document.createElement('input');
        this._searchElement.setAttribute('id', 'search');
        this._searchElement.setAttribute('type', 'text');
        this._searchElement.setAttribute('placeholder', 'Search...');
        this._searchElement.setAttribute('style', 'width: 100%');
        this._searchElement.addEventListener('input', (e) => {
            this.update(e.target.value);
            this.raise('search-text-changed', e.target.value);
        });
        this._resultElement = document.createElement('ol');
        this._resultElement.addEventListener('click', (e) => {
            this.select(e);
        });
        this._contentElement.appendChild(this._searchElement);
        this._contentElement.appendChild(this._resultElement);
    }

    on(event, callback) {
        this._events = this._events || {};
        this._events[event] = this._events[event] || [];
        this._events[event].push(callback);
    }

    raise(event, data) {
        if (this._events && this._events[event]) {
            this._events[event].forEach((callback) => {
                callback(this, data);
            });
        }
    }

    select(e) {
        var selection = [];
        var id = e.target.id;

        var nodesElement = this._graphElement.getElementById('nodes');
        var nodeElement = nodesElement.firstChild;
        while (nodeElement) { 
            if (nodeElement.id == id) {
                selection.push(nodeElement);
            }
            nodeElement = nodeElement.nextSibling;
        }

        var edgePathsElement = this._graphElement.getElementById('edge-paths');
        var edgePathElement = edgePathsElement.firstChild; 
        while (edgePathElement) {
            if (edgePathElement.id == id) {
                selection.push(edgePathElement);
            }
            edgePathElement = edgePathElement.nextSibling;
        }

        if (selection.length > 0) {
            this.raise('select', selection);
        }
    }

    focus(searchText) {
        this._searchElement.focus();
        this._searchElement.value = '';
        this._searchElement.value = searchText;
        this.update(searchText);
    }

    update(searchText) {
        while (this._resultElement.lastChild) {
            this._resultElement.removeChild(this._resultElement.lastChild);
        }

        var text = searchText.toLowerCase();

        var nodeMatches = {};
        var edgeMatches = {};

        this._graph.nodes.forEach((node) => {
            node.inputs.forEach((input) => {
                input.connections.forEach((connection) => {
                    if (connection.id && connection.id.toLowerCase().indexOf(text) != -1 && !edgeMatches[connection.id]) {
                        var item = document.createElement('li');
                        if (!connection.initializer) {
                            item.innerText = '\u2192 ' + connection.id.split('@').shift();
                            item.id = 'edge-' + connection.id;
                            this._resultElement.appendChild(item);
                            edgeMatches[connection.id] = true;
                        }
                    }    
                });
            });

            var name = node.name;
            if (name && name.toLowerCase().indexOf(text) != -1 && !nodeMatches[name]) {
                var item = document.createElement('li');
                item.innerText = '\u25A2 ' + node.name;
                item.id = 'node-' + node.name;
                this._resultElement.appendChild(item);
                nodeMatches[node.name] = true;
            }
        });

        this._graph.nodes.forEach((node) => {
            node.outputs.forEach((output) => {
                output.connections.forEach((connection) => {
                    if (connection.id && connection.id.toLowerCase().indexOf(text) != -1 && !edgeMatches[connection.id]) {
                        var item = document.createElement('li');
                        item.innerText = '\u2192 ' + connection.id.split('@').shift();
                        item.id = 'edge-' + connection.id;
                        this._resultElement.appendChild(item);
                        edgeMatches[connection.id] = true;
                    }    
                });
            });
        });

        this._resultElement.style.display = this._resultElement.childNodes.length != 0 ? 'block' : 'none';
    }
    
    get content() {
        return this._contentElement;
    }

}
