
const require = async () => {
    if (typeof process !== 'undefined' && process.versions && process.versions.node) {
        const worker_threads = await import('worker_threads');
        return worker_threads.isMainThread ? null : worker_threads.parentPort;
    }
    return typeof window !== 'undefined' && typeof self !== 'undefined' && self === window ? null : self;
};

require().then((self) => {
    self.addEventListener('message', async (e) => {
        const message = e.data;
        switch (message.type) {
            case 'dagre.layout': {
                const dagre = await import('./dagre.js');
                const graph = new dagre.Graph(true, message.compound);
                for (const node of message.nodes) {
                    graph.setNode(node.v, {
                        width: node.width,
                        height: node.height
                    });
                    if (node.parent) {
                        graph.setParent(node.v, node.parent);
                    }
                }
                for (const edge of message.edges) {
                    graph.setEdge(edge.v, edge.w, {
                        minlen: edge.minlen || 1,
                        weight: edge.weight || 1,
                        width: edge.width || 0,
                        height: edge.height || 0,
                        labeloffset: edge.labeloffset || 10,
                        labelpos: edge.labelpos || 'r'
                    });
                }
                dagre.layout(graph, message.layout);
                for (const node of message.nodes) {
                    const label = graph.node(node.v).label;
                    node.x = label.x;
                    node.y = label.y;
                    if (graph.children(node.v).length) {
                        node.width = label.width;
                        node.height = label.height;
                    }
                }
                for (const edge of message.edges) {
                    const label = graph.edge(edge.v, edge.w).label;
                    edge.points = label.points;
                    if ('x' in label) {
                        edge.x = label.x;
                        edge.y = label.y;
                    }
                }
                self.postMessage(message);
                break;
            }
            default: {
                throw Error(`Unsupported message type '${message.type}'.`);
            }
        }
    });
});

