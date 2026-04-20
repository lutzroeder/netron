
import * as fs from 'fs';
import * as path from 'path';
import * as playwright from '@playwright/test';
import * as url from 'url';

playwright.test.setTimeout(120000);

const getModelPath = (fileName) => {
    const self = url.fileURLToPath(import.meta.url);
    const dir = path.dirname(self);
    return path.resolve(dir, `../third_party/test/onnx/${fileName}`);
};

const openModel = async (page, modelPath) => {
    playwright.expect(fs.existsSync(modelPath)).toBeTruthy();

    const fileChooserPromise = page.waitForEvent('filechooser');
    const openButton = await page.locator('.open-file-button, button:has-text("Open Model")');
    await openButton.click();
    const fileChooser = await fileChooserPromise;
    await fileChooser.setFiles(modelPath);

    await page.waitForSelector('#canvas', { state: 'attached', timeout: 10000 });
    await page.waitForSelector('body.default', { timeout: 10000 });
};

const openFindSidebar = async (page) => {
    const menuButton = await page.locator('#menu-button');
    await menuButton.click();
    await page.waitForTimeout(200);
    const findMenuItem = await page.locator('button:has-text("Find...")');
    await findMenuItem.click();
    await page.waitForTimeout(500);
    const search = await page.waitForSelector('#search', { state: 'visible', timeout: 5000 });
    return search;
};

const getSearchInputValue = async (page) => {
    const searchInput = await page.locator('#search');
    return await searchInput.inputValue();
};

const getFindResultsCount = async (page) => {
    const results = await page.locator('.sidebar-find-content li');
    return await results.count();
};

playwright.test.describe('Find search state management', () => {

    playwright.test('scenario 1: switching models should clear search state', async ({ page }) => {
        const candyPath = getModelPath('candy.onnx');

        await page.goto('/');
        await page.waitForLoadState('domcontentloaded');
        await page.waitForSelector('body.welcome', { timeout: 25000 });
        await page.waitForTimeout(1000);

        const consent = await page.locator('#message-button');
        if (await consent.isVisible({ timeout: 25000 })) {
            await consent.click();
        }

        await openModel(page, candyPath);

        const search = await openFindSidebar(page);
        await search.fill('convolution');
        await page.waitForTimeout(500);

        const searchValueAfterSearch = await getSearchInputValue(page);
        playwright.expect(searchValueAfterSearch).toBe('convolution');

        const resultsCount = await getFindResultsCount(page);
        playwright.expect(resultsCount).toBeGreaterThan(0);

        await page.goto('/');
        await page.waitForLoadState('domcontentloaded');
        await page.waitForSelector('body.welcome', { timeout: 25000 });
        await page.waitForTimeout(1000);

        await openModel(page, candyPath);

        const search2 = await openFindSidebar(page);
        const searchValueAfterReopen = await getSearchInputValue(page);

        playwright.expect(searchValueAfterReopen).toBe('');
    });

    playwright.test('scenario 2: returning to welcome page and reopening model should clear search state', async ({ page }) => {
        const candyPath = getModelPath('candy.onnx');

        await page.goto('/');
        await page.waitForLoadState('domcontentloaded');
        await page.waitForSelector('body.welcome', { timeout: 25000 });
        await page.waitForTimeout(1000);

        const consent = await page.locator('#message-button');
        if (await consent.isVisible({ timeout: 25000 })) {
            await consent.click();
        }

        await openModel(page, candyPath);

        const search = await openFindSidebar(page);
        await search.fill('convolution1_W');
        await page.waitForSelector('.sidebar-find-content li', { state: 'attached' });

        const searchValueAfterSearch = await getSearchInputValue(page);
        playwright.expect(searchValueAfterSearch).toBe('convolution1_W');

        const resultsCount = await getFindResultsCount(page);
        playwright.expect(resultsCount).toBeGreaterThan(0);

        await page.goto('/');
        await page.waitForLoadState('domcontentloaded');
        await page.waitForSelector('body.welcome', { timeout: 25000 });
        await page.waitForTimeout(1000);

        await openModel(page, candyPath);

        const search2 = await openFindSidebar(page);
        const searchValueAfterReopen = await getSearchInputValue(page);

        playwright.expect(searchValueAfterReopen).toBe('');

        const resultsCountAfterReopen = await getFindResultsCount(page);
        playwright.expect(resultsCountAfterReopen).toBe(0);
    });

    playwright.test('scenario 3: search within same model should retain state when navigating subgraphs', async ({ page }) => {
        const candyPath = getModelPath('candy.onnx');

        await page.goto('/');
        await page.waitForLoadState('domcontentloaded');
        await page.waitForSelector('body.welcome', { timeout: 25000 });
        await page.waitForTimeout(1000);

        const consent = await page.locator('#message-button');
        if (await consent.isVisible({ timeout: 25000 })) {
            await consent.click();
        }

        await openModel(page, candyPath);

        const search = await openFindSidebar(page);
        await search.fill('convolution');
        await page.waitForTimeout(500);

        const searchValueBefore = await getSearchInputValue(page);
        playwright.expect(searchValueBefore).toBe('convolution');

        const backButton = await page.locator('#toolbar-path-back-button');
        const backButtonVisible = await backButton.isVisible();
        
        if (backButtonVisible) {
            const backButtonOpacity = await backButton.evaluate((el) => window.getComputedStyle(el).opacity);
            if (backButtonOpacity !== '0') {
                await backButton.click();
                await page.waitForTimeout(500);

                const searchValueAfter = await getSearchInputValue(page);
                playwright.expect(searchValueAfter).toBe('convolution');
            }
        }
    });

});
