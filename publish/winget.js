
const crypto = require('crypto');
const fs = require('fs');
const http = require('http');
const https = require('https');
const path = require('path');

const manifestDir = process.argv[2];
const configuration = require('../package.json');

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
const publisher = configuration.author.name;
const packageIdentifier = publisher.replace(' ', '') + '.' + productName;
const license = 'Copyright (c) ' + publisher;
const repository = 'https://github.com/' + configuration.repository;
const url = repository + '/releases/download/v' + version + '/' + productName + '-Setup-' + version + '.exe';
const extensions = configuration.build.fileAssociations.map((entry) => '- ' + entry.ext).sort().join('\n');

get(url).then((data) => {
    const sha256 = crypto.createHash('sha256').update(data).digest('hex').toUpperCase();
    const versionDir = path.join(manifestDir, publisher[0].toLowerCase(), publisher.replace(' ', ''), productName, version);
    if (!fs.existsSync(versionDir)){
        fs.mkdirSync(versionDir, { recursive: true });
    }
    const manifestFile = path.join(versionDir, packageIdentifier);
    fs.writeFileSync(manifestFile + '.yaml', [
        'PackageIdentifier: ' + packageIdentifier,
        'PackageVersion: ' + version,
        'DefaultLocale: en-US',
        'ManifestType: version',
        'ManifestVersion: 1.0.0',
        ''
    ].join('\n'));
    fs.writeFileSync(manifestFile + '.installer.yaml', [
        'PackageIdentifier: ' + packageIdentifier,
        'PackageVersion: ' + version,
        'Platform:',
        '- Windows.Desktop',
        'InstallModes:',
        '- silent',
        '- silentWithProgress',
        'Installers:',
        '- Architecture: x86',
        '  Scope: user',
        '  InstallerType: nullsoft',
        '  InstallerUrl: ' + url,
        '  InstallerSha256: ' + sha256,
        '  InstallerLocale: en-US',
        '  InstallerSwitches:',
        '    Custom: /NORESTART',
        '  UpgradeBehavior: install',
        '- Architecture: arm64',
        '  Scope: user',
        '  InstallerType: nullsoft',
        '  InstallerUrl: ' + url,
        '  InstallerSha256: ' + sha256,
        '  InstallerLocale: en-US',
        '  InstallerSwitches:',
        '    Custom: /NORESTART',
        '  UpgradeBehavior: install',
        'FileExtensions:',
        extensions,
        'ManifestType: installer',
        'ManifestVersion: 1.0.0',
        ''
    ].join('\n'));
    fs.writeFileSync(manifestFile + '.locale.en-US.yaml', [
        'PackageIdentifier: ' + packageIdentifier,
        'PackageVersion: ' + version,
        'PackageName: ' + productName,
        'PackageLocale: en-US',
        'PackageUrl: ' + repository,
        'Publisher: ' + publisher,
        'PublisherUrl: ' + repository,
        'PublisherSupportUrl: ' + repository + '/issues',
        'Author: ' + publisher,
        'License: ' + license,
        'Copyright: ' + license,
        'CopyrightUrl: ' + repository + '/blob/main/LICENSE',
        'ShortDescription: ' + configuration.description,
        'Description: ' + configuration.description,
        'Moniker: ' + name,
        'Tags:',
        '- machine-learning',
        '- deep-learning',
        '- neural-network',
        'ManifestType: defaultLocale',
        'ManifestVersion: 1.0.0',
        ''
    ].join('\n'));
}).catch((err) => {
    /* eslint-disable */
    console.log(err.message);
    /* eslint-enable */
});
