
const child_process = require('child_process');
const path = require('path');

const options = { stdio: 'inherit' };

const setupPath = path.join('publish', 'setup.py');

child_process.spawnSync('python', [ setupPath, '--quiet', 'build' ], options);

options.env = Object.assign({}, process.env);
options.env.PYTHONPATH = path.join('dist', 'lib');

const args = [ '-c', 'import netron; netron.main()' ].concat(process.argv.slice(2));
child_process.spawnSync('python', args, options);
