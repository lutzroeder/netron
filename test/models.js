
import * as fs from 'fs/promises';
import * as os from 'os';
import * as path from 'path';
import * as process from 'process';
import * as url from 'url';
import * as worker_threads from 'worker_threads';

const clearLine = () => {
    if (process.stdout.clearLine) {
        process.stdout.clearLine();
    }
};

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

const exit = (error) => {
    /* eslint-disable no-console */
    console.error(`${error.name}: ${error.message}`);
    if (error.cause) {
        console.error(`  ${error.cause.name}: ${error.cause.message}`);
    }
    /* eslint-enable no-console */
    process.exit(1);
};

const dirname = (...args) => {
    const file = url.fileURLToPath(import.meta.url);
    const dir = path.dirname(file);
    return path.join(dir, ...args);
};

const configuration = async () => {
    const file = dirname('models.json');
    const content = await fs.readFile(file, 'utf-8');
    return JSON.parse(content);
};

class Logger {

    constructor(threads) {
        this.threads = threads;
        this.entries = new Map();
    }

    update(identifier, message) {
        let value = null;
        if (message) {
            switch (message.name) {
                case 'name':
                    clearLine();
                    write(`${message.target}\n`);
                    value = '';
                    break;
                case 'download':
                    value = message.percent !== undefined ?
                        `${(`  ${Math.floor(100 * message.percent)}`).slice(-3)}% ` :
                        ` ${message.position}${this.threads === 1 ? ' bytes' : ''} `;
                    break;
                case 'decompress':
                    value = this.threads === 1 ? 'decompress' : '  ^  ';
                    break;
                case 'write':
                    value = this.threads === 1 ? 'write' : '  *  ';
                    break;
                default:
                    throw new Error(`Unsupported status message '${status.name}'.`);
            }
        }
        if (!this.entries.has(identifier) || this.entries.get(identifier) !== value) {
            this.entries.set(identifier, value);
            this.flush();
        }
    }

    delete(identifier) {
        this.entries.delete(identifier);
        this.flush();
    }

    flush() {
        clearLine();
        const values = Array.from(this.entries.values());
        if (!values.every((value) => !value)) {
            const list = values.map((value) => value || '     ');
            write(`  ${list.length > 0 ? list.join('-') : ''}\r`);
        }
    }
}

class Queue extends Array {

    constructor(targets, patterns) {
        if (patterns.length > 0) {
            patterns = patterns.map((pattern) => {
                const wildcard = pattern.indexOf('*') !== -1;
                return new RegExp(`^${wildcard ? `${pattern.replace(/\*/g, '.*')}$` : pattern}`);
            });
            targets = targets.filter((target) => {
                for (const file of target.target.split(',')) {
                    const value = target.type ? `${target.type}/${file}` : file;
                    if (patterns.some((pattern) => pattern.test(value))) {
                        return true;
                    }
                }
                return false;
            });
        }
        super(...targets.reverse());
    }
}

class Table {

    constructor(schema) {
        this.schema = schema;
        const line = `${Array.from(this.schema).join(',')}\n`;
        this.content = [ line ];
    }

    async add(row) {
        row = new Map(row);
        const line = `${Array.from(this.schema).map((key) => {
            const value = row.has(key) ? row.get(key) : '';
            row.delete(key);
            return value;
        }).join(',')}\n`;
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

class Worker {

    constructor(identifier, queue, logger, measures) {
        this._identifier = identifier;
        this._queue = queue;
        this._logger = logger;
        this._measures = measures;
    }

    async start() {
        this._events = {};
        this._events.message = (message) => this._message(message);
        this._events.error = (error) => this._error(error);
        this._worker = new worker_threads.Worker('./test/worker.js');
        for (let task = this._queue.pop(); task; task = this._queue.pop()) {
            this._logger.update(this._identifier, null);
            /* eslint-disable no-await-in-loop */
            await new Promise((resolve) => {
                this._resolve = resolve;
                this._attach();
                this._worker.postMessage(task);
            });
            /* eslint-enable no-await-in-loop */
        }
        this._logger.delete(this._identifier);
        await this._worker.terminate();
    }
    _attach() {
        this._worker.on('message', this._events.message);
        this._worker.on('error', this._events.error);
    }

    _detach() {
        this._worker.off('message', this._events.message);
        this._worker.off('error', this._events.error);
    }

    async _message(message) {
        switch (message.type) {
            case 'status': {
                this._logger.update(this._identifier, message);
                break;
            }
            case 'error': {
                write(`\n${message.target}\n`);
                this._error(message.error);
                break;
            }
            case 'complete': {
                await this._measures.add(message.measures);
                this._detach();
                this._resolve();
                delete this._resolve;
                break;
            }
            default: {
                throw new Error(`Unsupported message type '${message.type}'.`);
            }
        }
    }

    _error(error) {
        this._detach();
        delete this._resolve;
        exit(error);
    }
}

const main = async () => {
    try {
        const args = process.argv.length > 2 ? process.argv.slice(2) : [];
        const exists = await Promise.all(args.map((pattern) => access(pattern)));
        const paths = exists.length > 0 && exists.every((value) => value);
        const patterns = paths ? [] : args;
        const targets = paths ? args.map((path) => ({ target: path })) : await configuration();
        const queue = new Queue(targets, patterns);
        const threads = undefined;
        const logger = new Logger(threads);
        const measures = new Table([ 'name', 'download', 'load', 'validate', 'render' ]);
        // await measures.log(dirname('..', 'dist', 'test', 'measures.csv'));
        if (threads === 1) {
            const worker = await import('./worker.js');
            for (let item = queue.pop(); item; item = queue.pop()) {
                const target = new worker.Target(item);
                target.on('status', (_, message) => logger.update('', message));
                /* eslint-disable no-await-in-loop */
                await target.execute();
                await measures.add(target.measures);
                /* eslint-enable no-await-in-loop */
            }
        } else {
            const cores = Math.min(10, Math.round(0.7 * os.cpus().length), queue.length);
            const identifiers = [...new Array(cores).keys()].map((value) => value.toString());
            const workers = identifiers.map((identifier) => new Worker(identifier, queue, logger, measures));
            const promises = workers.map((worker) => worker.start());
            await Promise.all(promises);
        }
    } catch (error) {
        exit(error);
    }
};

main();
