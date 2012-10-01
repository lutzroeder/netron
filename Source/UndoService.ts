module Netron
{
	export class UndoService
	{
		private _container: ContainerUndoUnit = null;
		private _stack: ContainerUndoUnit[] = [];
		private _position: number = 0;

		public begin()
		{
			this._container = new ContainerUndoUnit();
		}

		public cancel()
		{
			this._container = null;
		}

		public commit()
		{
			if (!this._container.isEmpty)
			{
				this._stack.splice(this._position, this._stack.length - this._position);
				this._stack.push(this._container);
				this.redo();
			}
			this._container = null;	
		}

		public add(undoUnit: IUndoUnit)
		{
			this._container.add(undoUnit);
		}

		public undo()
		{
			if (this._position !== 0)
			{
				this._position--;
				this._stack[this._position].undo();
			}
		}

		public redo()
		{
			if ((this._stack.length !== 0) && (this._position < this._stack.length))
			{
				this._stack[this._position].redo();
				this._position++;
			}
		}
	}
}