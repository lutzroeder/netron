
const crypto = require('crypto');
const fs = require('fs');
const http = require('http');
const https = require('https');
const path = require('path');

const packageManifestFile = process.argv[2];
const manifestDir = process.argv[3];

const request = (url, timeout) => {
    return new Promise((resolve, reject) => {
        const httpModule = url.split(':').shift() === 'https' ? https : http;
        httpModule.get(url, (response) => {
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
                request(response.headers.location).then((data) => {
                    resolve(data);
                }).catch((err) => {
                    request(err);
                });
            }
            else {
                const err = new Error("The web request failed with status code " + response.statusCode + " at '" + url + "'.");
                err.type = 'error';
                err.url = url;
                err.status = response.statusCode;
                reject(err);
            }
        }).on("error", (err) => {
            reject(err);
        });
        if (timeout) {
            request.setTimeout(timeout, () => {
                request.abort();
                const err = new Error("The web request timed out at '" + url + "'.");
                err.type = 'timeout';
                err.url = url;
                reject(err);
            });
        }
    });
};

const packageManifest = JSON.parse(fs.readFileSync(packageManifestFile, 'utf-8'));
const name = packageManifest.name;
const version = packageManifest.version;
const productName = packageManifest.productName;
const publisher = packageManifest.author.name;
const repository = packageManifest.repository;
const url = 'https://github.com/' + repository + '/releases/download/v' + version + '/' + productName + '-Setup-' + version + '.exe';

request(url).then((data) => {
    const sha256 = crypto.createHash('sha256').update(data).digest('hex').toUpperCase();
    const lines = [
        'Id: ' + publisher.replace(' ', '') + '.' + productName,
        'Version: ' + version,
        'Name: ' + productName,
        'Publisher: ' + publisher,
        'AppMoniker: ' + name,
        'Description: ' + packageManifest.description,
        'License: Copyright (c) ' + publisher,
        'Homepage: ' + 'https://github.com/' + repository,
        'Installers:',
        '  - Arch: x86',
        '    InstallerType: nullsoft',
        '    Url: ' + url,
        '    Sha256: ' + sha256,
        ''
    ];
    const productDir = path.join(manifestDir, publisher.replace(' ', ''), productName);
    for (const file of fs.readdirSync(productDir)) {
        const versionFile = path.join(productDir, file);
        if (fs.lstatSync(versionFile).isFile()) {
            fs.unlinkSync(versionFile);
        }
    }
    const manifestFile = path.join(productDir, version + '.yaml');
    fs.writeFileSync(manifestFile, lines.join('\n'));
}).catch((err) => {
    console.log(err.message);
});
