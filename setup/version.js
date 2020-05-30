
const fs = require('fs');

const packageFile = process.argv[2];
const package = JSON.parse(fs.readFileSync(packageFile, 'utf-8'));
package.version = Array.from((parseInt(package.version.split('.').join(''), 10) + 1).toString()).join('.');
fs.writeFileSync(packageFile, JSON.stringify(package, null, 4) + '\n', 'utf-8');
