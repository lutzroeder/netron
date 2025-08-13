
import playwright from '@playwright/test';

export default playwright.defineConfig({
    outputDir: '../dist/test-results',
    reporter: './playwright.reporter.js',
    webServer: {
        command: 'npm run server',
        port: 8080,
        timeout: 120 * 1000
    },
    projects: [
        {
            name: 'desktop',
            testMatch: '**/desktop.spec.js',
        },
        {
            name: 'browser',
            testMatch: '**/browser.spec.js',
            use: {
                baseURL: 'http://localhost:8080'
            },
        },
    ],
});
