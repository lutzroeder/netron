
const child_process = require('child_process');
const fs = require('fs');
const path = require('path');

const source = path.join('source');
const target = path.join('dist', 'pypi', 'netron');

if (fs.existsSync(target)) {
    fs.rmSync(target, { recursive: true });
}

fs.mkdirSync(target, { recursive: true });

for (const entry of fs.readdirSync(source, { withFileTypes: true })) {
    if (entry.isFile()) {
        fs.copyFileSync(path.join(source, entry.name), path.join(target, entry.name));
    }
}

const options = { stdio: 'inherit' };
options.env = Object.assign({}, process.env);
options.env.PYTHONPATH = path.join('dist', 'pypi');

const args = [ '-c', 'import netron; netron.main()' ].concat(process.argv.slice(2));
child_process.spawnSync('python', args, options);
