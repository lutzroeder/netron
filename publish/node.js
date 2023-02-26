
const child_process = require('child_process');
const fs = require('fs');
const path = require('path');

const root = path.dirname(__dirname);

const write = (message) => {
    if (process.stdout.write) {
        process.stdout.write(message);
    }
};

const rm = (...args) => {
    write('rm ' + path.join(...args) + '\n');
    const dir = path.join(root, ...args);
    fs.rmSync(dir, { recursive: true, force: true });
};

const exec = (command) => {
    write(command + '\n');
    child_process.execSync(command, { cwd: root, stdio: [ 0,1,2 ] });
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
    exec('npx eslint source/*.js test/*.js publish/*.js tools/*.js');
    exec('python -m pip install --upgrade --quiet pylint');
    exec('python -m pylint -sn source/*.py publish/*.py test/*.py tools/*.py');
};

switch (process.argv[2]) {
    case 'install': clean(); break;
    case 'clean': clean(); break;
    case 'reset': reset(); break;
    case 'lint': lint(); break;
    default: break;
}
