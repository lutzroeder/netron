
const child_process = require('child_process');
const fs = require('fs');
const path = require('path');

const source = path.join('source');
const target = path.join('dist', 'pypi', 'netron');

if (fs.existsSync(target)) {
    fs.rmdirSync(target, { recursive: true });
}

fs.mkdirSync(target, { recursive: true });

for (const file of fs.readdirSync(source)) {
    fs.copyFileSync(path.join(source, file), path.join(target, file));
}

const options = { stdio: 'inherit' };
options.env = Object.assign({}, process.env);
options.env.PYTHONPATH = path.join('dist', 'pypi');

const args = [ '-c', 'import netron; netron.main()' ].concat(process.argv.slice(2));
child_process.spawnSync('python', args, options);
