@echo off

pushd %~dp0

rd /s /q ..\Build
md ..\Build\Debug
md ..\Build\Release

echo Building 'Release\*.html'
copy ..\Samples\demo_orgchart.html ..\Build\Debug\demo_orgchart.html

echo Building 'Debug\*.html'
copy ..\Samples\demo_orgchart.html ..\Build\Release\demo_orgchart.html

echo Building 'Debug\netron.js'
for %%i in ("..\Source\*.ts") do call set Source=%%Source%% ..\Source\%%i
node tsc.js -target ES5 -out ..\Build\Debug\netron.js lib.d.ts libex.ts %Source%

echo Building 'Release\netron.js'
node minify.js ..\Build\Debug\netron.js ..\Build\Release\netron.js

popd

echo Done.
