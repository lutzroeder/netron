
build: build_python build_electron

publish: clean publish_pip publish_github publish_cask

install:
	rm -rf ./node_modules
	npm install

clean:
	rm -rf ./build

build_python:
	@[ -d node_modules ] || npm install
	python ./setup.py build

build_electron:
	@[ -d node_modules ] || npm install
	npx electron-builder install-app-deps
	npx electron-builder --mac --linux --win

start:
	@[ -d node_modules ] || npm install
	npx electron .

publish_pip:
	@[ -d node_modules ] || npm install
	python ./setup.py build bdist_wheel upload

publish_github:
	@[ -d node_modules ] || npm install
	npx electron-builder install-app-deps
	npx electron-builder --mac --linux --win --publish always --draft false --prerelease false

publish_cask:
	@curl -H "Authorization: token $(GITHUB_TOKEN)" https://api.github.com/repos/caskroom/homebrew-cask/forks -d ''
	@export PACKAGE_VERSION=`node -pe "require('./package.json').version"`; \
	cask-repair --cask-version $$PACKAGE_VERSION --blind-submit netron
	@curl -H "Authorization: token $(GITHUB_TOKEN)" -X "DELETE" https://api.github.com/repos/lutzroeder/homebrew-cask
