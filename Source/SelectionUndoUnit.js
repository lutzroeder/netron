
SelectionUndoUnit = function()
{
	this.states = [];
}

SelectionUndoUnit.prototype.undo = function()
{
	for (var i = 0; i < this.states.length; i++)
	{
		if (this.states[i].undo)
		{
			this.states[i].value.select();
		}
		else
		{
			this.states[i].value.deselect();
		}
	}
};

SelectionUndoUnit.prototype.redo = function()
{
	for (var i = 0; i < this.states.length; i++)
	{
		if (this.states[i].redo)
		{
			this.states[i].value.select();
		}
		else
		{
			this.states[i].value.deselect();
		}
	}
};

SelectionUndoUnit.prototype.select = function(value)
{
	this.update(value, value.selected, true);
};

SelectionUndoUnit.prototype.deselect = function(value)
{
	this.update(value, value.selected, false);
};

SelectionUndoUnit.prototype.update = function(value, undo, redo)
{
	for (var i = 0; i < this.states.length; i++)
	{
		if (this.states[i].value == value)
		{
			this.states[i].redo = redo;
			return;
		}
	}
	this.states.push({ value: value, undo: undo, redo: redo });
};

SelectionUndoUnit.prototype.isEmpty = function()
{
	for (var i = 0; i < this.states.length; i++)
	{
		if (this.states[i].undo != this.states[i].redo)
		{
			return false;
		}
	}
	return true;
};
