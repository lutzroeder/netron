
DeleteConnectionUndoUnit = function(connection)
{
	this.connection = connection;
	this.from = connection.from;
	this.to = connection.to;
}

DeleteConnectionUndoUnit.prototype.undo = function()
{
	this.connection.insert(this.from, this.to);
};

DeleteConnectionUndoUnit.prototype.redo = function()
{
	this.connection.remove();
};
