
const child_process = require('child_process');
const fs = require('fs');
const notarize = require('electron-notarize');

exports.default = function (context) {
    if (context.electronPlatformName === 'darwin' && context.packager.platformSpecificBuildOptions.type !== 'development') {

        const appPath = context.appOutDir + '/' + context.packager.appInfo.productFilename + '.app';

        const configuration = fs.readFileSync('electron-builder.yml', 'utf-8');
        const appBundleId = (/^appId:\s(.*)\s/m.exec(configuration) || [ '', '' ])[1];

        const idResult = child_process.spawnSync('/usr/bin/security', [ 'find-generic-password', '-s', appBundleId, '-g' ], { encoding: 'utf-8' });
        const id = idResult.status === 0 ? (/"acct"<blob>="(.*)"/.exec(idResult.stdout) || [ '', ''])[1] : '';

        const passwordResult = child_process.spawnSync('/usr/bin/security', [ 'find-generic-password', '-s', appBundleId, '-w' ], { encoding: 'utf-8' });
        const password = passwordResult.status == 0 ? passwordResult.stdout.split('\n').shift() : '';

        return notarize.notarize({
            appBundleId: appBundleId,
            appPath: appPath,
            appleId: id,
            appleIdPassword: password,
        });
    }
};
