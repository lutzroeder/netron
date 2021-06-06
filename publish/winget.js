
const crypto = require('crypto');
const fs = require('fs');
const http = require('http');
const https = require('https');
const path = require('path');

const packageManifestFile = process.argv[2];
const manifestDir = process.argv[3];

const get = (url, timeout) => {
    return new Promise((resolve, reject) => {
        const httpModule = url.split(':').shift() === 'https' ? https : http;
        const request = httpModule.request(url, {}, (response) => {
            if (response.statusCode === 200) {
                const data = [];
                let position = 0;
                response.on('data', (chunk) => {
                    data.push(chunk);
                    position += chunk.length;
                    process.stdout.write('  ' + position + ' bytes\r');
                });
                response.on('err', (err) => {
                    reject(err);
                });
                response.on('end', () => {
                    resolve(Buffer.concat(data));
                });
            }
            else if (response.statusCode === 302) {
                get(response.headers.location).then((data) => {
                    resolve(data);
                }).catch((err) => {
                    reject(err);
                });
            }
            else {
                const err = new Error("The web request failed with status code " + response.statusCode + " at '" + url + "'.");
                err.type = 'error';
                err.url = url;
                err.status = response.statusCode;
                reject(err);
            }
        });
        request.on("error", (err) => {
            reject(err);
        });
        if (timeout) {
            request.setTimeout(timeout, () => {
                request.destroy();
                const err = new Error("The web request timed out at '" + url + "'.");
                err.type = 'timeout';
                err.url = url;
                reject(err);
            });
        }
        request.end();
    });
};

const packageManifest = JSON.parse(fs.readFileSync(packageManifestFile, 'utf-8'));
const name = packageManifest.name;
const version = packageManifest.version;
const productName = packageManifest.productName;
const publisher = packageManifest.author.name;
const repository = packageManifest.repository;
const url = 'https://github.com/' + repository + '/releases/download/v' + version + '/' + productName + '-Setup-' + version + '.exe';

get(url).then((data) => {
    const sha256 = crypto.createHash('sha256').update(data).digest('hex').toUpperCase();
    const lines = [
        'PackageIdentifier: ' + publisher.replace(' ', '') + '.' + productName,
        'PackageVersion: ' + version,
        'PackageName: ' + productName,
        'Publisher: ' + publisher,
        'Moniker: ' + name,
        'ShortDescription: ' + packageManifest.description,
        'License: Copyright (c) ' + publisher,
        'PackageUrl: ' + 'https://github.com/' + repository,
        'Installers:',
        '- Architecture: x86',
        '  InstallerType: nullsoft',
        '  InstallerUrl: ' + url,
        '  InstallerSha256: ' + sha256,
        'PackageLocale: en-US',
        'ManifestType: singleton',
        'ManifestVersion: 1.0.0',
        ''
    ];
    const versionDir = path.join(manifestDir, publisher[0].toLowerCase(), publisher.replace(' ', ''), productName, version);
    if (!fs.existsSync(versionDir)){
        fs.mkdirSync(versionDir);
    }
    const manifestFile = path.join(versionDir, publisher.replace(' ', '') + '.' + productName + '.yaml');
    fs.writeFileSync(manifestFile, lines.join('\n'));
}).catch((err) => {
    console.log(err.message);
});
