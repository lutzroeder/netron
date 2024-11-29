
/*
const APPLE_API_KEY_ID = process.env.APPLE_API_KEY_ID;
const APPLE_API_KEY_ISSUER_ID = process.env.APPLE_API_KEY_ISSUER_ID;
*/

export default {
    outDir: 'dist',
    packagerConfig: {
        icon: 'publish/icon',
        dir: [
            'source'
        ],
        ignore: [
            'publish',
            'third_party',
            'test',
            'tools'
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
                // name: '${name}-${version}-mac.zip'
            }
        },
        {
            name: '@electron-forge/maker-dmg',
            config: {
                background: './publish/background.png',
                /* eslint-disable no-template-curly-in-string */
                name: 'Netron-${version}'
                /* eslint-enable no-template-curly-in-string */
            }
        },
        {
            // sudo snap install snapcraft --classic --channel=7.x/stable
            // DEBUG=electron-installer-snap:snapcraft npx electron-forge make --arch x64 --targets=@electron-forge/maker-snap
            // DEBUG=electron-installer-snap:snapcraft npx electron-forge make --arch arm64 --targets=@electron-forge/maker-snap
            // Using "DEBUG=electron-installer-snap:snapcraft" will enable --destructive-mode and skip multipass
            // To unpack the .snap: unsquashfs dist/make/snap/arm64/netron_8.0.0_arm64.snap
            name: '@electron-forge/maker-snap',
            platforms: ['linux'],
            config: {
                grade: 'stable',
                // base: 'core20',
                apps: {
                    electronApp: {
                        plugs: [
                            'removable-media'
                        ]
                    }
                }
            }
        }
    ],
    publishers: [
        {
            'name': '@electron-forge/publisher-github',
            'config': {}
        },
        {
            'name': '@electron-forge/publisher-snapcraft',
            'config': {
                release: 'latest/stable'
            }
        }
    ]
};
