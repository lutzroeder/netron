module Netron
{
	export class ContainerUndoUnit implements IUndoUnit
	{
		private _undoUnits: IUndoUnit[] = [];

		public add(undoUnit: IUndoUnit)
		{
			this._undoUnits.push(undoUnit);
		}

		public undo()
		{
			for (var i = 0; i < this._undoUnits.length; i++)
			{
				this._undoUnits[i].undo();
			}
		}

		public redo()
		{
			for (var i = 0; i < this._undoUnits.length; i++)
			{
				this._undoUnits[i].redo();
			}
		}

		public get isEmpty(): boolean
		{
			if (this._undoUnits.length > 0)
			{
				for (var i = 0; i < this._undoUnits.length; i++)
				{
					if (!this._undoUnits[i].isEmpty)
					{
						return false;
					}
				}
			}
			return true;
		}
	}
}