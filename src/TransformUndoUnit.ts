module Netron
{
    export class TransformUndoUnit implements IUndoUnit
    {
        private _element: Element;
        private _undoRectangle: Rectangle;
        private _redoRectangle: Rectangle;

        constructor(element: Element, undoRectangle: Rectangle, redoRectangle: Rectangle)
        {
            this._element = element;
            this._undoRectangle = undoRectangle.clone();
            this._redoRectangle = redoRectangle.clone();
        }

        public undo()
        {
            this._element.rectangle = this._undoRectangle;
        }

        public redo()
        {
            this._element.rectangle = this._redoRectangle;
        }

        public get isEmpty(): boolean
        {
            return false;
        }
    }
}