
InsertElementUndoUnit = function(element, owner)
{
	this.element = element;
	this.owner = owner;
}

InsertElementUndoUnit.prototype.undo = function()
{
	this.element.remove();
};

InsertElementUndoUnit.prototype.redo = function()
{
	this.element.insertInto(this.owner);
};
