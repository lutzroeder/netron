
.PHONY: test coverage

build: clean lint build_python build_electron

publish: clean lint publish_electron publish_python publish_web publish_cask publish_winget

install:
	@[ -d node_modules ] || npm install

clean:
	rm -rf ./dist
	rm -rf ./node_modules
	rm -rf ./package-lock.json

reset: clean
	rm -rf ./third_party/env
	rm -rf ./third_party/source

update: install
	@./tools/armnn sync schema
	@./tools/bigdl sync schema
	@./tools/caffe sync schema
	@./tools/circle sync schema metadata
	@./tools/cntk sync schema
	@./tools/coreml sync schema
	@./tools/dlc schema
	@./tools/dnn schema
	@./tools/mnn sync schema
	@./tools/mslite sync schema metadata
	@./tools/megengine sync schema metadata
	@./tools/nnabla sync schema metadata
	@./tools/onnx sync install schema metadata
	@./tools/om schema
	@./tools/rknn schema
	@./tools/paddle sync schema
	@./tools/pytorch sync schema metadata
	@./tools/sklearn sync install metadata
	@./tools/tf sync install schema metadata
	@./tools/uff schema
	@./tools/xmodel sync schema

build_python: install
	python publish/python.py build version
	python -m pip install --user build wheel --quiet
	python -m build --no-isolation --wheel --outdir ./dist/pypi dist/pypi

install_python: build_python
	pip install --force-reinstall dist/pypi/*.whl

build_electron: install
	npx electron-builder --mac --universal --publish never -c.mac.identity=null
	npx electron-builder --win --x64 --arm64 --publish never
	npx electron-builder --linux appimage --x64 --publish never
	npx electron-builder --linux snap --x64 --publish never

start: install
	npx electron .

lint: install
	npx eslint source/*.js test/*.js publish/*.js tools/*.js
	python -m pip install --upgrade --quiet pylint onnx torch torchvision
	python -m pylint -sn source/*.py publish/*.py test/*.py tools/*.py

codeql:
	@[ -d third_party/tools/codeql ] || git clone --depth=1 https://github.com/github/codeql.git ./third_party/tools/codeql
	rm -rf dist/codeql
	mkdir -p dist/codeql/netron
	cp -r publish source test tools dist/codeql/netron/
	codeql database create dist/codeql/database --source-root dist/codeql/netron --language=javascript --threads=3
	codeql database analyze dist/codeql/database ./third_party/tools/codeql/javascript/ql/src/codeql-suites/javascript-security-and-quality.qls --format=csv --output=dist/codeql/results.csv --threads=3
	cat dist/codeql/results.csv

test: install
	node ./test/models.js

coverage:
	rm -rf .nyc_output ./coverage ./dist/nyc
	mkdir -p ./dist/nyc
	cp ./package.json ./dist/nyc
	cp -R ./source ./dist/nyc
	nyc instrument --compact false ./source ./dist/nyc/source
	nyc --reporter=lcov --instrument npx electron ./dist/nyc

publish_python: build_python
	python -m pip install --user twine
	python -m twine upload --non-interactive --skip-existing --verbose dist/pypi/*.whl

publish_electron: install
	npx electron-builder --mac --universal --publish always
	npx electron-builder --win --x64 --arm64 --publish always
	npx electron-builder --linux appimage --x64 --publish always
	npx electron-builder --linux snap --x64 --publish always

build_web:
	mkdir -p ./dist/web
	rm -rf ./dist/web/*
	cp -R ./source/*.html ./dist/web
	cp -R ./source/*.css ./dist/web
	cp -R ./source/*.js ./dist/web
	cp -R ./source/*.json ./dist/web
	cp -R ./source/*.ico ./dist/web
	cp -R ./source/*.png ./dist/web
	rm -rf ./dist/web/electron.* ./dist/web/app.js
	node ./publish/web.js ./package.json ./dist/web/index.html

publish_web: build_web
	rm -rf ./dist/gh-pages
	git clone --depth=1 https://x-access-token:$(GITHUB_TOKEN)@github.com/$(GITHUB_USER)/netron.git --branch gh-pages ./dist/gh-pages 2>&1 > /dev/null
	cp -R ./dist/web/* ./dist/gh-pages
	git -C ./dist/gh-pages add --all
	git -C ./dist/gh-pages commit --amend --no-edit
	git -C ./dist/gh-pages push --force origin gh-pages

publish_cask:
	curl -s -H "Authorization: token ${GITHUB_TOKEN}" -X "DELETE" https://api.github.com/repos/${GITHUB_USER}/homebrew-cask 2>&1 > /dev/null
	sleep 4
	curl -s -H "Authorization: token $(GITHUB_TOKEN)" https://api.github.com/repos/Homebrew/homebrew-cask/forks -d '' 2>&1 > /dev/null
	rm -rf ./dist/homebrew-cask
	sleep 4
	git clone --depth=2 https://x-access-token:$(GITHUB_TOKEN)@github.com/$(GITHUB_USER)/homebrew-cask.git ./dist/homebrew-cask
	node ./publish/cask.js ./dist/homebrew-cask/Casks/netron.rb
	git -C ./dist/homebrew-cask add --all
	git -C ./dist/homebrew-cask commit -m "Update $$(node -pe "require('./package.json').productName") to $$(node -pe "require('./package.json').version")"
	git -C ./dist/homebrew-cask push
	curl -H "Authorization: token $(GITHUB_TOKEN)" https://api.github.com/repos/Homebrew/homebrew-cask/pulls -d "{\"title\":\"Update $$(node -pe "require('./package.json').name") to $$(node -pe "require('./package.json').version")\",\"base\":\"master\",\"head\":\"$(GITHUB_USER):master\",\"body\":\"Update version and sha256.\"}" 2>&1 > /dev/null
	rm -rf ./dist/homebrew-cask

publish_winget:
	curl -s -H "Authorization: token ${GITHUB_TOKEN}" -X "DELETE" https://api.github.com/repos/${GITHUB_USER}/winget-pkgs 2>&1 > /dev/null
	sleep 4
	curl -s -H "Authorization: token $(GITHUB_TOKEN)" https://api.github.com/repos/microsoft/winget-pkgs/forks -d '' 2>&1 > /dev/null
	rm -rf ./dist/winget-pkgs
	sleep 4
	git clone --depth=2 https://x-access-token:$(GITHUB_TOKEN)@github.com/$(GITHUB_USER)/winget-pkgs.git ./dist/winget-pkgs
	node ./publish/winget.js ./dist/winget-pkgs/manifests
	git -C ./dist/winget-pkgs add --all
	git -C ./dist/winget-pkgs commit -m "Update $$(node -pe "require('./package.json').name") to $$(node -pe "require('./package.json').version")"
	git -C ./dist/winget-pkgs push
	curl -H "Authorization: token $(GITHUB_TOKEN)" https://api.github.com/repos/microsoft/winget-pkgs/pulls -d "{\"title\":\"Update $$(node -pe "require('./package.json').productName") to $$(node -pe "require('./package.json').version")\",\"base\":\"master\",\"head\":\"$(GITHUB_USER):master\",\"body\":\"\"}" 2>&1 > /dev/null
	rm -rf ./dist/winget-pkgs

version:
	node ./publish/version.js ./package.json
	git add ./package.json
	git commit -m "Update to $$(node -pe "require('./package.json').version")"
	git tag v$$(node -pe "require('./package.json').version")
	git push
	git push --tags

pull:
	git fetch --prune origin "refs/tags/*:refs/tags/*"
	git pull --prune --rebase
