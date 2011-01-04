
ContainerUndoUnit = function()
{
	this.undoUnits = [];
}

ContainerUndoUnit.prototype.add = function(undoUnit)
{
	this.undoUnits.push(undoUnit);
};

ContainerUndoUnit.prototype.undo = function()
{
	for (var i = 0; i < this.undoUnits.length; i++)
	{
		this.undoUnits[i].undo();
	}
};

ContainerUndoUnit.prototype.redo = function()
{
	for (var i = 0; i < this.undoUnits.length; i++)
	{
		this.undoUnits[i].redo();
	}
};

ContainerUndoUnit.prototype.isEmpty = function()
{
	if (this.undoUnits.length > 0)
	{
		for (var i = 0; i < this.undoUnits.length; i++)
		{
			if (!("isEmpty" in this.undoUnits[i]) || !this.undoUnits[i].isEmpty())
			{
				return false;
			}
		}
	}
	return true;
};
