
const require = async () => {
    if (typeof process !== 'undefined' && process.versions && process.versions.node) {
        const worker_threads = await import('worker_threads');
        return worker_threads.parentPort;
    }
    return self;
};

require().then((self) => {
    self.addEventListener('message', async (e) => {
        const message = e.data;
        switch (message.type) {
            case 'dagre.layout': {
                const dagre = await import('./dagre.js');
                dagre.layout(message.nodes, message.edges, message.layout, {});
                self.postMessage(message);
                break;
            }
            default: {
                throw Error(`Unsupported message type '${message.type}'.`);
            }
        }
    });
});

