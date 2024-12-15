
import * as fs from 'fs/promises';
import * as inspector from 'inspector';
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
    } catch {
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
        this._threads = threads;
        this._entries = new Map();
    }

    update(identifier, message) {
        let value = null;
        if (message) {
            switch (message.name) {
                case 'name':
                    delete this._cache;
                    clearLine();
                    write(`${message.target}\n`);
                    value = '';
                    break;
                case 'download':
                    if (message.percent !== undefined) {
                        value = `${(`  ${Math.floor(100 * message.percent)}`).slice(-3)}% `;
                    } else if (Number.isInteger(message.position)) {
                        value = ` ${message.position}${this._threads === 1 ? ' bytes' : ''} `;
                    } else {
                        value = '  \u2714  ';
                    }
                    break;
                case 'decompress':
                    value = this._threads === 1 ? 'decompress' : '  ^  ';
                    break;
                case 'write':
                    value = this._threads === 1 ? 'write' : '  *  ';
                    break;
                default:
                    throw new Error(`Unsupported status message '${message.name}'.`);
            }
        }
        if (!this._entries.has(identifier) || this._entries.get(identifier) !== value) {
            this._entries.set(identifier, value);
            this._flush();
        }
    }

    delete(identifier) {
        this._entries.delete(identifier);
        this._flush();
    }

    flush() {
        delete this._cache;
        this._flush();
    }

    _flush() {
        const values = Array.from(this._entries.values());
        const text = values.some((s) => s) ? `  ${values.map((s) => s || '     ').join('-')}\r` : '';
        if (this._cache !== text) {
            this._cache = text;
            clearLine();
            write(text);
        }
    }
}

class Queue extends Array {

    constructor(targets, patterns) {
        for (const target of targets) {
            target.targets = target.target.split(',');
            target.name = target.type ? `${target.type}/${target.targets[0]}` : target.targets[0];
            target.tags = target.tags ? target.tags.split(',') : [];
        }
        if (patterns.length > 0) {
            const tags = new Set();
            patterns = patterns.filter((pattern) => {
                if (pattern.startsWith('tag:')) {
                    tags.add(pattern.substring(4));
                    return false;
                }
                return true;
            });
            patterns = patterns.map((pattern) => {
                const wildcard = pattern.indexOf('*') !== -1;
                return new RegExp(`^${wildcard ? `${pattern.replace(/\*/g, '.*')}$` : pattern}`);
            });
            targets = targets.filter((target) => {
                for (const file of target.targets) {
                    const value = target.type ? `${target.type}/${file}` : file;
                    if (patterns.some((pattern) => pattern.test(value))) {
                        return true;
                    }
                    if (target.tags.some((tag) => tags.has(tag))) {
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
        this.content = [line];
        this.entries = [];
    }

    async add(row) {
        this.entries.push(row);
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

    summarize(name) {
        const entries = this.entries.filter((entry) => entry.has(name));
        return entries.map((entry) => entry.get(name)).reduce((a, c) => a + c, 0);
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
            task.measures = this._measures ? new Map() : null;
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
                if (this._measures) {
                    await this._measures.add(message.measures);
                }
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
        const args = { inputs: [], measure: false, profile: false };
        if (process.argv.length > 2) {
            for (const arg of process.argv.slice(2)) {
                switch (arg) {
                    case 'measure': args.measure = true; break;
                    case 'profile': args.profile = true; break;
                    default: args.inputs.push(arg); break;
                }
            }
        }
        const exists = await Promise.all(args.inputs.map((pattern) => access(pattern)));
        const paths = exists.length > 0 && exists.every((value) => value);
        const patterns = paths ? [] : args.inputs;
        const targets = paths ? args.inputs.map((path) => ({ target: path, tags: 'quantization,validation' })) : await configuration();
        const queue = new Queue(targets, patterns);
        const threads = args.measure || inspector.url() ? 1 : undefined;
        const logger = new Logger(threads);
        let measures = null;
        if (args.measure) {
            measures = new Table(['name', 'download', 'load', 'validate', 'render']);
            await measures.log(dirname('..', 'dist', 'test', 'measures.csv'));
        }
        let session = null;
        if (args.profile) {
            session = new inspector.Session();
            session.connect();
            await new Promise((resolve, reject) => {
                session.post('Profiler.enable', (error) => error ? reject(error) : resolve());
            });
            await new Promise((resolve, reject) => {
                session.post('Profiler.start', (error) => error ? reject(error) : resolve());
            });
            /* eslint-disable no-console */
            console.profile();
            /* eslint-enable no-console */
        }
        if (threads === 1) {
            const worker = await import('./worker.js');
            for (let item = queue.pop(); item; item = queue.pop()) {
                const target = new worker.Target(item);
                target.measures = measures ? new Map() : null;
                target.on('status', (_, message) => logger.update('', message));
                /* eslint-disable no-await-in-loop */
                await target.execute();
                if (target.measures) {
                    await measures.add(target.measures);
                }
                /* eslint-enable no-await-in-loop */
            }
        } else {
            const threads = Math.min(10, Math.round(0.7 * os.cpus().length), queue.length);
            const identifiers = [...new Array(threads).keys()].map((value) => value.toString());
            const workers = identifiers.map((identifier) => new Worker(identifier, queue, logger, measures));
            const promises = workers.map((worker) => worker.start());
            await Promise.all(promises);
        }
        if (args.measure) {
            const values = {
                download: measures.summarize('download'),
                load: measures.summarize('load'),
                validate: measures.summarize('validate'),
                render: measures.summarize('render')
            };
            values.total = values.load + values.validate + values.render;
            const pad1 = Math.max(...Object.keys(values).map((key) => key.length));
            const pad2 = Math.max(...Object.values(values).map((value) => value.toFixed(2).indexOf('.')));
            write('\n');
            for (let [key, value] of Object.entries(values)) {
                key = `${key}:`.padEnd(pad1 + 1);
                value = `${value.toFixed(2)}`.padStart(pad2 + 3);
                write(`${key} ${value}\n`);
            }
            write('\n');
        }
        if (args.profile) {
            /* eslint-disable no-console */
            console.profileEnd();
            /* eslint-enable no-console */
            const data = await new Promise((resolve, reject) => {
                session.post('Profiler.stop', (error, data) => error ? reject(error) : resolve(data));
            });
            session.disconnect();
            const file = dirname('..', 'dist', 'test', 'profile.cpuprofile');
            await fs.mkdir(path.dirname(file), { recursive: true });
            await fs.writeFile(file, JSON.stringify(data.profile), 'utf-8');
        }
    } catch (error) {
        exit(error);
    }
};

await main();
