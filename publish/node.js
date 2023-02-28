
const child_process = require('child_process');
const fs = require('fs');
const os = require('os');
const path = require('path');

const root = path.dirname(__dirname);

const write = (message) => {
    if (process.stdout.write) {
        process.stdout.write(message + os.EOL);
    }
};

const rm = (...args) => {
    write('rm ' + path.join(...args));
    const dir = path.join(root, ...args);
    fs.rmSync(dir, { recursive: true, force: true });
};

const exec = (command) => {
    try {
        child_process.execSync(command, { cwd: root, stdio: [ 0,1,2 ] });
    } catch (error) {
        process.exit(1);
    }
};

const install = () => {
    if (!fs.existsSync(path.join(root, 'node_modules'))) {
        child_process.execSync('npm install', { cwd: root, stdio: [ 0,1,2 ] });
    }
};

const clean = () => {
    rm('dist');
    rm('node_modules');
    rm('package-lock.json');
};

const reset = () => {
    clean();
    rm('third_party', 'env');
    rm('third_party', 'source');
};

const lint = () => {
    install();
    write('eslint');
    exec('npx eslint source/*.js test/*.js publish/*.js tools/*.js');
    write('pylint');
    exec('python -m pip install --upgrade --quiet pylint');
    exec('python -m pylint -sn source/*.py publish/*.py test/*.py tools/*.py');
};

const update = () => {
    const targets = process.argv.length > 3 ? process.argv.slice(3) : [
        'armnn',
        'bigdl',
        'caffe',
        'circle',
        'cntk',
        'coreml',
        'dlc',
        'dnn',
        'mnn',
        'mslite',
        'megengine',
        'nnabla',
        'onnx',
        'om',
        'paddle',
        'pytorch',
        'rknn',
        'sklearn',
        'tf',
        'uff',
        'xmodel'
    ];
    for (const target of targets) {
        exec('tools/' + target + ' sync install schema metadata');
    }
};

const pull = () => {
    exec('git fetch --prune origin "refs/tags/*:refs/tags/*"');
    exec('git pull --prune --rebase');
};

switch (process.argv[2]) {
    case 'install': clean(); break;
    case 'clean': clean(); break;
    case 'reset': reset(); break;
    case 'lint': lint(); break;
    case 'update': update(); break;
    case 'pull': pull(); break;
    default: break;
}
