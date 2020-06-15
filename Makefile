
.PHONY: test

build: clean lint build_python build_electron

publish: clean lint publish_electron publish_python publish_github_pages publish_cask publish_winget

install:
	@[ -d node_modules ] || npm install

clean:
	rm -rf ./dist
	rm -rf ./node_modules
	rm -rf ./package-lock.json

reset: clean
	rm -rf ./third_party

update: install
	@./tools/armnn sync schema
	@./tools/bigdl sync schema
	@./tools/caffe sync schema
	@./tools/cntk sync schema
	@./tools/coreml sync schema
	@./tools/mnn sync schema
	@./tools/onnx sync install schema metadata
	@./tools/paddle sync schema
	@./tools/pytorch sync install schema metadata
	@./tools/sklearn sync install metadata
	@./tools/tf sync install schema metadata
	@./tools/uff schema

build_python: install
	python -m pip install --user wheel
	python ./setup.py build --version bdist_wheel

build_electron: install
	CSC_IDENTITY_AUTO_DISCOVERY=false npx electron-builder --mac --publish never
	npx electron-builder --win --publish never
	npx electron-builder --linux appimage --publish never
	npx electron-builder --linux snap --publish never

lint: install
	npx eslint src/*.js test/*.js setup/*.js tools/*.js

test: install
	node ./test/test.js

start: install
	npx electron .

publish_python: build_python
	python -m pip install --user twine
	python -m twine upload --non-interactive --skip-existing --verbose dist/dist/*

publish_electron: install
	npx electron-builder --mac --publish always
	npx electron-builder --win --publish always
	npx electron-builder --linux appimage --publish always
	npx electron-builder --linux snap --publish always

publish_github_pages: build_python
	rm -rf ./dist/gh-pages
	git clone --depth=1 https://x-access-token:$(GITHUB_TOKEN)@github.com/$(GITHUB_USER)/netron.git --branch gh-pages ./dist/gh-pages 2>&1 > /dev/null
	rm -rf ./dist/gh-pages/*
	cp -R ./dist/lib/netron/* ./dist/gh-pages/
	rm -rf ./dist/gh-pages/*.py*
	git -C ./dist/gh-pages add --all
	git -C ./dist/gh-pages commit --amend --no-edit
	git -C ./dist/gh-pages push --force origin gh-pages

publish_cask:
	curl -s -H "Authorization: token $(GITHUB_TOKEN)" https://api.github.com/repos/Homebrew/homebrew-cask/forks -d '' 2>&1 > /dev/null
	rm -rf ./dist/homebrew-cask
	sleep 4
	git clone --depth=2 https://x-access-token:$(GITHUB_TOKEN)@github.com/$(GITHUB_USER)/homebrew-cask.git ./dist/homebrew-cask
	node ./setup/cask.js ./package.json ./dist/homebrew-cask/Casks/netron.rb
	git -C ./dist/homebrew-cask add --all
	git -C ./dist/homebrew-cask commit -m "Update $$(node -pe "require('./package.json').productName") to $$(node -pe "require('./package.json').version")"
	git -C ./dist/homebrew-cask push
	curl -H "Authorization: token $(GITHUB_TOKEN)" https://api.github.com/repos/Homebrew/homebrew-cask/pulls -d "{\"title\":\"Add $$(node -pe "require('./package.json').name") $$(node -pe "require('./package.json').version")\",\"base\":\"master\",\"head\":\"$(GITHUB_USER):master\",\"body\":\"\"}" 2>&1 > /dev/null
	rm -rf ./dist/homebrew-cask
	curl -s -H "Authorization: token $(GITHUB_TOKEN)" -X "DELETE" https://api.github.com/repos/$(GITHUB_USER)/homebrew-cask 2>&1 > /dev/null

publish_winget:
	curl -s -H "Authorization: token $(GITHUB_TOKEN)" https://api.github.com/repos/microsoft/winget-pkgs/forks -d '' 2>&1 > /dev/null
	rm -rf ./dist/winget-pkgs
	sleep 4
	git clone --depth=2 https://x-access-token:$(GITHUB_TOKEN)@github.com/$(GITHUB_USER)/winget-pkgs.git ./dist/winget-pkgs
	node ./setup/winget.js ./package.json ./dist/winget-pkgs/manifests
	git -C ./dist/winget-pkgs add --all
	git -C ./dist/winget-pkgs commit -m "Update $$(node -pe "require('./package.json').name") to $$(node -pe "require('./package.json').version")"
	git -C ./dist/winget-pkgs push
	curl -H "Authorization: token $(GITHUB_TOKEN)" https://api.github.com/repos/microsoft/winget-pkgs/pulls -d "{\"title\":\"Add $$(node -pe "require('./package.json').productName") $$(node -pe "require('./package.json').version")\",\"base\":\"master\",\"head\":\"$(GITHUB_USER):master\",\"body\":\"\"}" 2>&1 > /dev/null
	rm -rf ./dist/winget-pkgs
	curl -s -H "Authorization: token $(GITHUB_TOKEN)" -X "DELETE" https://api.github.com/repos/$(GITHUB_USER)/winget-pkgs 2>&1 > /dev/null

version:
	node ./setup/version.js ./package.json
	git add ./package.json
	git commit -m "Update to $$(node -pe "require('./package.json').version")"
	git tag v$$(node -pe "require('./package.json').version")
	git push --force
	git push --tags
	git tag -d v$$(node -pe "require('./package.json').version")
