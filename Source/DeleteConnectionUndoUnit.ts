module Netron
{
	export class DeleteConnectionUndoUnit implements IUndoUnit
	{
		private _connection: Connection;
		private _from: Connector;
		private _to: Connector;

		constructor(connection: Connection)
		{
			this._connection = connection;
			this._from = connection.from;
			this._to = connection.to;
		}

		public undo()
		{
			this._connection.insert(this._from, this._to);
		}

		public redo()
		{
			this._connection.remove();
		}

		public get isEmpty(): bool
		{
			return false;
		}
	}
}