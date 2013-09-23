module Netron
{
	export class ContentChangedUndoUnit implements IUndoUnit
	{
		private _element: Element;
		private _undoContent: any;
		private _redoContent: any;

		constructor(element: Element, content: any)
		{
			this._element = element;
			this._undoContent = element.content;
			this._redoContent = content;
		}

		public undo()
		{
			this._element.content = this._undoContent;
		}

		public redo()
		{
			this._element.content = this._redoContent;
		}

		public get isEmpty(): boolean
		{
			return false;
		}
	}
}