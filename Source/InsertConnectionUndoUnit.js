
function InsertConnectionUndoUnit(connection, from, to)
{
	this.connection = connection;
	this.from = from;
	this.to = to;
}

InsertConnectionUndoUnit.prototype.undo = function()
{
	this.connection.remove();
};

InsertConnectionUndoUnit.prototype.redo = function()
{
	this.connection.insert(this.from, this.to);
};
