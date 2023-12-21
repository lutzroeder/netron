
import * as fs from 'fs/promises';
import * as os from 'os';
import * as path from 'path';
import * as url from 'url';
import * as worker_threads from 'worker_threads';

const write = (message) => {
    if (process.stdout.write) {
        process.stdout.write(message);
    }
};

const access = async (path) => {
    try {
        await fs.access(path);
        return true;
    } catch (error) {
        return false;
    }
};

class Queue {

    constructor(targets, patterns) {
        this.targets = targets;
        this.patterns = patterns;
    }

    next() {
        while (this.targets.length > 0) {
            const item = this.targets.pop();
            if (this.patterns.length === 0) {
                return item;
            }
            const parts = item.target.split(',');
            const files = item.type ? parts : parts.map((file) => path.resolve(process.cwd(), file));
            const type = item.type;
            for (const pattern of this.patterns) {
                for (const file of files) {
                    const name = type + '/' + file;
                    const match = pattern.indexOf('*') !== -1 ?
                        new RegExp('^' + pattern.replace('*', '.*') + '$').test(name) :
                        name.startsWith(pattern);
                    if (match) {
                        return item;
                    }
                }
            }
        }
        return null;
    }
}

class Table {

    constructor(schema) {
        this.schema = schema;
        const line = Array.from(this.schema).join(',') + '\n';
        this.content = [ line ];
    }

    async add(row) {
        row = new Map(row);
        const line = Array.from(this.schema).map((key) => {
            const value = row.has(key) ? row.get(key) : '';
            row.delete(key);
            return value;
        }).join(',') + '\n';
        if (row.size > 0) {
            throw new Error();
        }
        this.content.push(line);
        if (this.file) {
            await fs.appendFile(this.file, line);
        }
    }

    async log(file) {
        if (file) {
            await fs.mkdir(path.dirname(file), { recursive: true });
            await fs.writeFile(file, this.content.join(''));
            this.file = file;
        }
    }
}

const main = async () => {
    try {
        let patterns = process.argv.length > 2 ? process.argv.slice(2) : [];
        const dirname = path.dirname(url.fileURLToPath(import.meta.url));
        const configuration = await fs.readFile(dirname + '/models.json', 'utf-8');
        let targets = JSON.parse(configuration).reverse();
        if (patterns.length > 0) {
            const exists = await Promise.all(patterns.map((pattern) => access(pattern)));
            if (exists.every((value) => value)) {
                targets = patterns.map((path) => {
                    return { target: path };
                });
                patterns = [];
            }
        }
        const queue = new Queue(targets, patterns);
        const measures = new Table([ 'name', 'download', 'load', 'validate', 'render' ]);
        await measures.log(path.join(dirname, '..', 'dist', 'test', 'measures.csv'));
        const mode = '';
        // const mode = 'threads';
        switch (mode) {
            case 'threads': {
                const workers = new Set();
                const cpus = Math.min(6, os.cpus().length - 2);
                for (let i = 0; i < cpus; i++) {
                    const worker = new worker_threads.Worker('./test/worker.js');
                    worker.next = function() {
                        const item = queue.next();
                        if (item) {
                            this.postMessage(item);
                        } else {
                            this.terminate();
                            workers.delete(this);
                            if (workers.size === 0) {
                                process.exit(1);
                            }
                        }
                    };
                    worker.on('message', async (message) => {
                        await measures.add(message.measures);
                        if (message.__error__) {
                            write(message.error);
                            process.exit(1);
                        } else {
                            // write(message.name);
                        }
                        worker.next();
                    });
                    workers.add(worker);
                    worker.next();
                }
                break;
            }
            default: {
                const worker = await import('./worker.js');
                const __host__ = await worker.Target.start();
                for (let item = queue.next(); item; item = queue.next()) {
                    const target = new worker.Target(__host__, item);
                    /* eslint-disable no-await-in-loop */
                    await target.execute();
                    await measures.add(target.measures);
                    /* eslint-enable no-await-in-loop */
                }
                break;
            }
        }
    } catch (error) {
        /* eslint-disable no-console */
        console.error(error.name + ': ' + error.message);
        if (error.cause) {
            console.error('  ' + error.cause.name + ': ' + error.cause.message);
        }
        /* eslint-enable no-console */
        process.exit(1);
    }
};

main();