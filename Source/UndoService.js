
var UndoService = function()
{
	this.container = null;
	this.stack = [];
	this.position = 0;
};

UndoService.prototype.begin = function()
{
	this.container = new ContainerUndoUnit();
};

UndoService.prototype.cancel = function()
{
	this.container = null;
};

UndoService.prototype.commit = function()
{
	if (!this.container.isEmpty())
	{
		this.stack.splice(this.position, this.stack.length - this.position);
		this.stack.push(this.container);
		this.redo();
	}
	this.container = null;	
};

UndoService.prototype.add = function(undoUnit)
{
	this.container.add(undoUnit);
};

UndoService.prototype.undo = function()
{
	if (this.position !== 0)
	{
		this.position--;
		this.stack[this.position].undo();
	}
};

UndoService.prototype.redo = function()
{
	if ((this.stack.length !== 0) && (this.position < this.stack.length))
	{
		this.stack[this.position].redo();
		this.position++;
	}
};
