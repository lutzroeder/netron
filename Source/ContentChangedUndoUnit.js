
ContentChangedUndoUnit = function(element, content)
{
	this.element = element;
	this.undoContent = element.content;
	this.redoContent = content;
}

ContentChangedUndoUnit.prototype.undo = function()
{
	this.element.content = this.undoContent;
};

ContentChangedUndoUnit.prototype.redo = function()
{
	this.element.content = this.redoContent;
};
