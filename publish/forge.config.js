
/*
const APPLE_API_KEY_ID = process.env.APPLE_API_KEY_ID;
const APPLE_API_KEY_ISSUER_ID = process.env.APPLE_API_KEY_ISSUER_ID;
*/

export default {
    outDir: 'dist',
    packagerConfig: {
        icon: "publish/icon",
        dir: [
            'source'
        ],
        ignore: [
            "publish",
            "third_party",
            "test",
            "tools"
        ],
        /*
        osxNotarize: {
            tool: 'notarytool',
            appleApiKey: `~/.private_keys/AuthKey_${APPLE_API_KEY_ID}.p8`,
            appleApiKeyId: APPLE_API_KEY_ID,
            appleApiIssuer: APPLE_API_KEY_ISSUER_ID
        },
        */
        asar: true
    },
    /*
    makeTargets: {
        win32: ['nsis'],
        darwin: ['dmg', 'zip'],
        linux: ['snap'],
    },
    */
    makers: [
        {
            name: '@electron-forge/maker-zip',
            config: {
                platforms: ['darwin'],
                // name: "${name}-${version}-mac.zip"
            }
        },
        {
            name: '@electron-forge/maker-dmg',
            config: {
                background: './publish/background.png',
                /* eslint-disable no-template-curly-in-string */
                name: "Netron-${version}"
                /* eslint-enable no-template-curly-in-string */
            }
        }
    ],
    publishers: [
        {
            "name": "@electron-forge/publisher-github",
            "config": {}
        },
        {
            "name": "@electron-forge/publisher-snapcraft",
            "config": {}
        }
    ]
};
