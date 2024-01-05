
const notarize = require('@electron/notarize');

exports.default = function (context) {
    if (process.platform === 'darwin' && context.electronPlatformName === 'darwin') {
        const config = context.packager.info.options.config;
        if (process.env.CSC_IDENTITY_AUTO_DISCOVERY !== 'false' && (!config || !config.mac || config.mac.identity !== null)) {
            const appId = context.packager.info.config.appId;
            const appOutDir = context.appOutDir;
            const productFilename = context.packager.appInfo.productFilename;
            const APPLE_API_KEY_ID = process.env.APPLE_API_KEY_ID;
            const APPLE_API_KEY_ISSUER_ID = process.env.APPLE_API_KEY_ISSUER_ID;
            return notarize.notarize({
                tool: 'notarytool',
                appBundleId: appId,
                appPath: `${appOutDir}/${productFilename}.app`,
                appleApiKey: `~/.private_keys/AuthKey_${APPLE_API_KEY_ID}.p8`,
                appleApiKeyId: APPLE_API_KEY_ID,
                appleApiIssuer: APPLE_API_KEY_ISSUER_ID
            });
        }
    }
    return null;
};
