
build: build_python build_electron

publish: clean publish_github_electron publish_pip publish_github_pages publish_cask

install:
	rm -rf ./node_modules
	npm install

clean:
	rm -rf ./build

build_python:
	@[ -d node_modules ] || npm install
	rm -rf ./build/python
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
	rm -rf ./build/python
	python ./setup.py build bdist_wheel upload

publish_github_electron:
	@[ -d node_modules ] || npm install
	npx electron-builder install-app-deps
	npx electron-builder --mac --linux --win --publish always

publish_github_pages:
	@[ -d node_modules ] || npm install
	python ./setup.py build
	rm -rf ./build/gh-pages
	git clone git@github.com:lutzroeder/Netron.git ./build/gh-pages --branch gh-pages
	rm -rf ./build/gh-pages/*
	cp -R ./build/python/lib/netron/* ./build/gh-pages/
	rm -rf ./build/gh-pages/*.py
	rm -rf ./build/gh-pages/*.pyc
	rm -rf ./build/gh-pages/netron
	mv ./build/gh-pages/view-browser.html ./build/gh-pages/index.html
	@export PACKAGE_VERSION=`node -pe "require('./package.json').version"`; \
	sed -i -e "s/<!-- meta -->/<meta name='version' content='$$PACKAGE_VERSION' \/>/g" ./build/gh-pages/index.html
	git -C ./build/gh-pages add --all
	git -C ./build/gh-pages commit --amend --no-edit
	git -C ./build/gh-pages push --force origin gh-pages

publish_cask:
	@curl -H "Authorization: token $(GITHUB_TOKEN)" https://api.github.com/repos/Homebrew/homebrew-cask/forks -d ''
	@export PACKAGE_VERSION=`node -pe "require('./package.json').version"`; \
	cask-repair --cask-version $$PACKAGE_VERSION --blind-submit netron
	@curl -H "Authorization: token $(GITHUB_TOKEN)" -X "DELETE" https://api.github.com/repos/lutzroeder/homebrew-cask
