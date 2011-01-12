
var TransformUndoUnit = function(element, undoRectangle, redoRectangle)
{
	this.element = element;
	this.undoRectangle = new Rectangle(undoRectangle.x, undoRectangle.y, undoRectangle.width, undoRectangle.height);
	this.redoRectangle = new Rectangle(redoRectangle.x, redoRectangle.y, redoRectangle.width, redoRectangle.height);
};

TransformUndoUnit.prototype.undo = function()
{
	this.element.setRectangle(this.undoRectangle);
};

TransformUndoUnit.prototype.redo = function()
{
	this.element.setRectangle(this.redoRectangle);
};
