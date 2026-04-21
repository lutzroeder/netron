
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

const hoverOverFirstSearchResult = async (page) => {
    const firstResult = await page.locator('.sidebar-find-content li').first();
    if (await firstResult.isVisible()) {
        await firstResult.hover();
        await page.waitForTimeout(300);
        return true;
    }
    return false;
};

const clickFirstSearchResult = async (page) => {
    const firstResult = await page.locator('.sidebar-find-content li').first();
    if (await firstResult.isVisible()) {
        await firstResult.click();
        await page.waitForTimeout(300);
        return true;
    }
    return false;
};

const dblClickFirstSearchResult = async (page) => {
    const firstResult = await page.locator('.sidebar-find-content li').first();
    if (await firstResult.isVisible()) {
        await firstResult.dblclick();
        await page.waitForTimeout(300);
        return true;
    }
    return false;
};

const getFirstSearchResultText = async (page) => {
    const firstResult = await page.locator('.sidebar-find-content li').first();
    if (await firstResult.isVisible()) {
        return await firstResult.textContent();
    }
    return null;
};

const isBackButtonEnabled = async (page) => {
    const backButton = await page.locator('#toolbar-path-back-button');
    const isVisible = await backButton.isVisible();
    if (!isVisible) {
        return false;
    }
    const opacity = await backButton.evaluate((el) => window.getComputedStyle(el).opacity);
    return opacity !== '0';
};

const clickBackButton = async (page) => {
    const backButton = await page.locator('#toolbar-path-back-button');
    await backButton.click();
    await page.waitForTimeout(500);
};

const searchAndVerify = async (page, searchTerm, expectedResults, expectedValue) => {
    const search = await openFindSidebar(page);
    await search.fill(searchTerm);
    await page.waitForTimeout(500);

    const searchValue = await getSearchInputValue(page);
    playwright.expect(searchValue).toBe(expectedValue || searchTerm);

    const resultsCount = await getFindResultsCount(page);
    if (expectedResults > 0) {
        playwright.expect(resultsCount).toBeGreaterThan(0);
    } else {
        playwright.expect(resultsCount).toBe(0);
    }

    return resultsCount;
};

playwright.test.describe('Find search state management', () => {

    playwright.test('scenario 1: switching from model A to model B should clear search state and highlights', async ({ page }) => {
        const candyPath = getModelPath('candy.onnx');
        const convAutopadPath = getModelPath('conv_autopad.onnx');

        const candyExists = fs.existsSync(candyPath);
        const convAutopadExists = fs.existsSync(convAutopadPath);

        if (!candyExists || !convAutopadExists) {
            console.log('Skipping test: test models not found');
            return;
        }

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

        const firstResultText = await getFirstSearchResultText(page);
        playwright.expect(firstResultText).not.toBeNull();

        await hoverOverFirstSearchResult(page);
        await clickFirstSearchResult(page);
        await page.waitForTimeout(200);

        await page.goto('/');
        await page.waitForLoadState('domcontentloaded');
        await page.waitForSelector('body.welcome', { timeout: 25000 });
        await page.waitForTimeout(1000);

        await openModel(page, convAutopadPath);

        const search2 = await openFindSidebar(page);
        const searchValueAfterSwitch = await getSearchInputValue(page);

        playwright.expect(searchValueAfterSwitch).toBe('');

        const resultsCountAfterSwitch = await getFindResultsCount(page);
        playwright.expect(resultsCountAfterSwitch).toBe(0);

        await search2.fill('convolution');
        await page.waitForTimeout(500);

        const newResultsCount = await getFindResultsCount(page);
        const newFirstResultText = await getFirstSearchResultText(page);

        if (newResultsCount > 0) {
            playwright.expect(newFirstResultText).not.toBe(firstResultText);
        }

        await search2.fill('');
        await page.waitForTimeout(200);
        const searchValueAfterClear = await getSearchInputValue(page);
        playwright.expect(searchValueAfterClear).toBe('');
    });

    playwright.test('scenario 2: returning to welcome page and reopening model should clear search state and highlights', async ({ page }) => {
        const candyPath = getModelPath('candy.onnx');

        if (!fs.existsSync(candyPath)) {
            console.log('Skipping test: test model not found');
            return;
        }

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

        const firstResultText = await getFirstSearchResultText(page);
        playwright.expect(firstResultText).not.toBeNull();

        await hoverOverFirstSearchResult(page);
        await dblClickFirstSearchResult(page);
        await page.waitForTimeout(300);

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

        await search2.fill('convolution1_W');
        await page.waitForTimeout(500);

        const newResultsCount = await getFindResultsCount(page);
        playwright.expect(newResultsCount).toBeGreaterThan(0);

        const newFirstResultText = await getFirstSearchResultText(page);
        playwright.expect(newFirstResultText).toBe(firstResultText);
    });

    playwright.test('scenario 3: search within same model with subgraph navigation should retain state', async ({ page }) => {
        const ifK1Path = getModelPath('if_k1.onnx');

        if (!fs.existsSync(ifK1Path)) {
            console.log('Skipping test: test model not found');
            return;
        }

        await page.goto('/');
        await page.waitForLoadState('domcontentloaded');
        await page.waitForSelector('body.welcome', { timeout: 25000 });
        await page.waitForTimeout(1000);

        const consent = await page.locator('#message-button');
        if (await consent.isVisible({ timeout: 25000 })) {
            await consent.click();
        }

        await openModel(page, ifK1Path);

        const search = await openFindSidebar(page);
        await search.fill('If');
        await page.waitForTimeout(500);

        const searchValueBefore = await getSearchInputValue(page);
        playwright.expect(searchValueBefore).toBe('If');

        const resultsCountBefore = await getFindResultsCount(page);
        playwright.expect(resultsCountBefore).toBeGreaterThan(0);

        const firstResultText = await getFirstSearchResultText(page);
        playwright.expect(firstResultText).not.toBeNull();

        const hasSubgraph = await isBackButtonEnabled(page);
        playwright.expect(hasSubgraph).toBe(true);

        await hoverOverFirstSearchResult(page);
        await dblClickFirstSearchResult(page);
        await page.waitForTimeout(300);

        const searchValueAfterNav = await getSearchInputValue(page);
        playwright.expect(searchValueAfterNav).toBe('If');

        const resultsCountAfterNav = await getFindResultsCount(page);
        playwright.expect(resultsCountAfterNav).toBeGreaterThan(0);

        const backButtonEnabled = await isBackButtonEnabled(page);
        playwright.expect(backButtonEnabled).toBe(true);

        await clickBackButton(page);

        const searchValueAfterBack = await getSearchInputValue(page);
        playwright.expect(searchValueAfterBack).toBe('If');

        const resultsCountAfterBack = await getFindResultsCount(page);
        playwright.expect(resultsCountAfterBack).toBeGreaterThan(0);

        const finalFirstResultText = await getFirstSearchResultText(page);
        playwright.expect(finalFirstResultText).toBe(firstResultText);
    });

});
