module Netron
{
    export class InsertConnectionUndoUnit implements IUndoUnit
    {   
        private _connection: Connection;
        private _from: Connector;
        private _to: Connector;

        constructor(connection: Connection, from: Connector, to: Connector)
        {
            this._connection = connection;
            this._from = from;
            this._to = to;
        }

        public undo()
        {
            this._connection.remove();
        }

        public redo()
        {
            this._connection.insert(this._from, this._to);
        }

        public get isEmpty(): boolean
        {
            return false;
        }
    }
}