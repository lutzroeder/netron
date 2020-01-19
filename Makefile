
.PHONY: test

build: clean lint build_python build_electron

publish: clean lint publish_github_electron publish_python publish_github_pages publish_cask

install:
	rm -rf ./node_modules
	rm -rf ./package-lock.json
	npm install

clean:
	rm -rf ./dist

reset:
	rm -rf ./dist
	rm -rf ./node_modules
	rm -rf ./third_party
	rm -rf ./package-lock.json

update:
	@[ -d node_modules ] || npm install
	@./tools/armnn sync schema
	@./tools/bigdl sync schema
	@./tools/caffe sync schema
	@./tools/coreml sync schema
	@./tools/chainer sync
	@./tools/cntk sync schema
	@./tools/darknet sync
	@./tools/dl4j sync
	@./tools/keras sync install metadata
	@./tools/mlnet sync metadata
	@./tools/mnn sync schema
	@./tools/mxnet sync metadata
	@./tools/ncnn sync
	@./tools/onnx sync install schema metadata
	@./tools/paddle sync schema
	@./tools/pytorch sync install schema metadata
	@./tools/sklearn sync install metadata
	@./tools/tf sync install schema metadata
	@./tools/tflite sync schema
	@./tools/torch sync

build_python:
	@[ -d node_modules ] || npm install
	python3 ./setup.py build --version

build_electron:
	@[ -d node_modules ] || npm install
	npx electron-builder install-app-deps
	npx electron-builder --mac
	npx electron-builder --win
	npx electron-builder --linux appimage
	npx electron-builder --linux snap

lint:
	@[ -d node_modules ] || npm install
	npx eslint src/*.js test/*.js

test:
	@[ -d node_modules ] || npm install
	node ./test/test.js

start:
	@[ -d node_modules ] || npm install
	npx electron .

publish_python:
	@[ -d node_modules ] || npm install
	python3 ./setup.py build --version bdist_wheel
	python3 -m pip install --user keyring
	python3 -m pip install --user twine
	twine upload dist/dist/*

publish_github_electron:
	@[ -d node_modules ] || npm install
	npx electron-builder install-app-deps
	npx electron-builder --mac --publish always
	npx electron-builder --win --publish always
	npx electron-builder --linux appimage --publish always
	npx electron-builder --linux snap --publish always

publish_github_pages:
	@[ -d node_modules ] || npm install
	python3 ./setup.py build --version
	rm -rf ./dist/gh-pages
	git clone git@github.com:lutzroeder/netron.git ./dist/gh-pages --branch gh-pages
	rm -rf ./dist/gh-pages/*
	cp -R ./dist/lib/netron/* ./dist/gh-pages/
	rm -rf ./dist/gh-pages/*.py*
	@export PACKAGE_VERSION=`node -pe "require('./package.json').version"`; \
	sed -i -e "s/<!-- meta -->/<meta name='version' content='$$PACKAGE_VERSION' \/>/g" ./dist/gh-pages/index.html
	git -C ./dist/gh-pages add --all
	git -C ./dist/gh-pages commit --amend --no-edit
	git -C ./dist/gh-pages push --force origin gh-pages

publish_cask:
	@curl -H "Authorization: token $(GITHUB_TOKEN)" https://api.github.com/repos/Homebrew/homebrew-cask/forks -d ''
	@export PACKAGE_VERSION=`node -pe "require('./package.json').version"`; \
	cask-repair --cask-version $$PACKAGE_VERSION --blind-submit netron
	@curl -H "Authorization: token $(GITHUB_TOKEN)" -X "DELETE" https://api.github.com/repos/lutzroeder/homebrew-cask
