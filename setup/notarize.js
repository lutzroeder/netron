const child_process = require('child_process');
const fs = require('fs');
const notarize = require('electron-notarize');

exports.default = function (context) {
    if (process.platform === 'darwin' && context.electronPlatformName === 'darwin') {
        const config = context.packager.info.options.config;
        const skip = config && config.mac && config.mac.identity === null;
        if (!skip) {
            return notarize.notarize({
                appBundleId: context.packager.info.config.appId,
                appPath: context.appOutDir + '/' + context.packager.appInfo.productFilename + '.app',
                appleApiKey: process.env.API_KEY_ID,
                appleApiIssuer: process.env.API_KEY_ISSUER_ID,
            });
        }
    }
}