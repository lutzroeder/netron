module Netron
{
    export class InsertElementUndoUnit implements IUndoUnit
    {
        private _element: Element;
        private _graph: Graph;

        constructor(element: Element, graph: Graph)
        {
            this._element = element;
            this._graph = graph;
        }

        public undo()
        {
            this._element.remove();
        }

        public redo()
        {
            this._element.insertInto(this._graph);
        }

        public get isEmpty(): boolean
        {
            return false;
        }
    }
}