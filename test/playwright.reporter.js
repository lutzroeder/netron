
class Reporter {

    onTestBegin(test) {
        process.stdout.write(`${test.title}\n`);
    }

    onTestEnd(test, result) {
        if (result.status === 'failed') {
            process.stderr.write(`\nTest failed: ${test.title}\n`);
            if (result.error) {
                process.stderr.write(`${result.error.message}\n`);
                if (result.error.stack) {
                    process.stderr.write(`${result.error.stack}\n`);
                }
            }
        }
    }

    onError(error) {
        process.stderr.write(`\nError: ${error}\n`);
    }
}

export default Reporter;