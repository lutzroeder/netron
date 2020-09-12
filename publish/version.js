
const fs = require('fs');

const packageManifestFile = process.argv[2];
const packageManifest = JSON.parse(fs.readFileSync(packageManifestFile, 'utf-8'));
packageManifest.version = Array.from((parseInt(packageManifest.version.split('.').join(''), 10) + 1).toString()).join('.');
fs.writeFileSync(packageManifestFile, JSON.stringify(packageManifest, null, 4) + '\n', 'utf-8');
