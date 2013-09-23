module Netron
{
	export class DeleteElementUndoUnit implements IUndoUnit
	{
		private _element: Element;
		private _graph: Graph;

		constructor(element: Element)
		{
			this._element = element;
			this._graph = element.graph;
		}

		public undo()
		{
			this._element.insertInto(this._graph);
		}

		public redo()
		{
			this._element.remove();
		}

		public get isEmpty(): boolean
		{
			return false;
		}
	}
}