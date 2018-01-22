
PACKAGE_VERSION=`node -pe "require('./package.json').version"`

dist: dist_electron dist_pip

publish: clean dist publish_pip publish_github publish_cask

install:
	@rm -rf node_modules
	@npm install
	@npx electron-builder install-app-deps

clean:
	@rm -rf dist
	@rm -rf build

dist_electron:
	@npx electron-builder install-app-deps
	@npx electron-builder --mac --linux --win

dist_pip:
	@python setup.py build bdist_wheel

start:
	@npx electron .

publish_pip:
	@python setup.py upload

publish_github:
	export GH_TOKEN=$(GITHUB_TOKEN);
	npx electron-builder install-app-deps
	npx electron-builder --publish always --draft false --prerelease false

publish_cask:
	curl -H "Authorization: token $(GITHUB_TOKEN)" https://api.github.com/repos/caskroom/homebrew-cask/forks -d ''
	cask-repair --cask-version $(PACKAGE_VERSION) --blind-submit netron
	curl -H "Authorization: token $(GITHUB_TOKEN)" -X "DELETE" https://api.github.com/repos/lutzroeder/homebrew-cask
