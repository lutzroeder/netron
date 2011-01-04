
DeleteElementUndoUnit = function(element)
{
	this.element = element;
	this.owner = this.element.owner;
}

DeleteElementUndoUnit.prototype.undo = function()
{
	this.element.insertInto(this.owner);
};

DeleteElementUndoUnit.prototype.redo = function()
{
	this.element.remove();
};
