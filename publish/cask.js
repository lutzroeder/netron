
const fs = require('fs');
const path = require('path');
const http = require('http');
const https = require('https');
const crypto = require('crypto');

const configuration = require('../package.json');
const caskFile = process.argv[2];

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

const name = configuration.name;
const version = configuration.version;
const productName = configuration.productName;
const description = configuration.description;
const repository = 'https://github.com/' + configuration.repository;
const url = repository + '/releases/download/v#{version}/' + productName + '-#{version}-mac.zip';
const location = url.replace(/#{version}/g, version);

get(location).then((data) => {
    const sha256 = crypto.createHash('sha256').update(data).digest('hex').toLowerCase();
    const caskDir = path.dirname(caskFile);
    if (!fs.existsSync(caskDir)){
        fs.mkdirSync(caskDir, { recursive: true });
    }
    fs.writeFileSync(caskFile, [
        'cask "' + name + '" do',
        '  version "' + version + '"',
        '  sha256 "' + sha256 + '"',
        '',
        '  url "' + url + '"',
        '  name "' + productName + '"',
        '  desc "' + description + '"',
        '  homepage "' + repository + '"',
        '',
        '  auto_updates true',
        '',
        '  app "' + productName + '.app"',
        'end',
        ''
    ].join('\n'));
}).catch((err) => {
    /* eslint-disable */
    console.log(err.message);
    /* eslint-enable */
});
