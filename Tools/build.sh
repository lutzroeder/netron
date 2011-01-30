#!/bin/bash

rm -r ../Build
mkdir ../Build
mkdir ../Build/Debug
mkdir ../Build/Release

cd ../Samples
cp demo_web.html ../Build/Debug/demo_web.html
cp demo_web.html ../Build/Release/demo_web.html
cp demo_genealogy.html ../Build/Debug/demo_genealogy.html
cp demo_genealogy.html ../Build/Release/demo_genealogy.html

echo Building \'Debug/netron.js\'.
cd ../Source
cat	Function.js \
	Array.js \
	Point.js \
	Rectangle.js \
	CanvasRenderingContext2D.js \
	Cursors.js \
	Connector.js \
	Tracker.js \
	Element.js \
	Connection.js \
	Selection.js \
	ContainerUndoUnit.js \
	InsertElementUndoUnit.js \
	DeleteElementUndoUnit.js \
	InsertConnectionUndoUnit.js \
	DeleteConnectionUndoUnit.js \
	ContentChangedUndoUnit.js \
	TransformUndoUnit.js \
	SelectionUndoUnit.js \
	UndoService.js \
	Graph.js \
	> ../Build/Debug/netron.js

cd ../Tools
echo Building \'Release/netron.js\'.
java -jar compiler.jar --js ../Build/Debug/netron.js > ../Build/Release/netron.js

echo Done.
