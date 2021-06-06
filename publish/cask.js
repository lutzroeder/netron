

const fs = require('fs');
const http = require('http');
const https = require('https');
const crypto = require('crypto');

const packageManifestFile = process.argv[2];
const caskFile = process.argv[3];

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
const description = packageManifest.description;
const repository = 'https://github.com/' + packageManifest.repository;
const url = repository + '/releases/download/v#{version}/' + productName + '-#{version}-mac.zip';
const location = url.replace(/#{version}/g, version);

get(location).then((data) => {
    const sha256 = crypto.createHash('sha256').update(data).digest('hex').toLowerCase();
    const lines = [
        'cask "' + name + '" do',
        '  version "' + version + '"',
        '  sha256 "' + sha256 + '"',
        '',
        '  url "' + url + '"',
        '  name "' + productName + '"',
        '  desc "' + description + '"',
        '  homepage "' + repository + '"',
        '',
        '  livecheck do',
        '    url :url',
        '    strategy :git',
        '    regex(/^v(\\d+(?:\\.\\d+)*)$/)',
        '  end',
        '',
        '  auto_updates true',
        '',
        '  app "' + productName + '.app"',
        'end',
        ''
    ];
    fs.writeFileSync(caskFile, lines.join('\n'));

}).catch((err) => {
    console.log(err.message);
});
