
function Connector(owner, template)
{
	this.owner = owner;
	this.template = template;
	this.connections = [];
	this.hover = false;
}

Connector.prototype.getCursor = function(point)
{
	return Cursors.grip;
};

Connector.prototype.hitTest = function(rectangle)
{
	if ((rectangle.width === 0) && (rectangle.height === 0))
	{
		return this.getRectangle().contains(rectangle.topLeft());
	}
	return rectangle.contains(this.getRectangle());
};

Connector.prototype.getRectangle = function()
{
	var point = this.owner.getConnectorPosition(this);
	var rectangle = new Rectangle(point.x, point.y, 0, 0);
	rectangle.inflate(3, 3);
	return rectangle;
};

Connector.prototype.invalidate = function()
{
};

Connector.prototype.paint = function(context)
{
	var rectangle = this.getRectangle();

	var strokeStyle = this.owner.owner.style.connectorBorder; 
	var fillStyle = this.owner.owner.style.connector;
	if (this.hover) // TODO || (this.owner.owner.newConnection !== null))
	{
		strokeStyle = this.owner.owner.style.connectorHoverBorder; 
		fillStyle = this.owner.owner.style.connectorHover;
		// if ((this.list) || (this.connections.Count != 1))
		// {
		//	fillColor = Color.FromArgb(0, 192, 0); // medium green
		// }
	}

	context.lineWidth = 1;
	context.strokeStyle = strokeStyle;
	context.lineCap = "butt";
	context.fillStyle = fillStyle;
	context.fillRect(rectangle.x - 0.5, rectangle.y - 0.5, rectangle.width, rectangle.height);
	context.strokeRect(rectangle.x - 0.5, rectangle.y - 0.5, rectangle.width, rectangle.height);

	if (this.hover)
	{
		// Tooltip
		var text = ("description" in this.template) ? this.template.description : this.template.name;
		context.textBaseline = "bottom";
		context.font = "8.25pt Tahoma";
		var size = context.measureText(text);
		size.height = 14;
		var a = new Rectangle(rectangle.x - Math.floor(size.width / 2), rectangle.y + size.height + 6, size.width, size.height);
		var b = new Rectangle(a.x, a.y, a.width, a.height);
		a.inflate(4, 1);
		context.fillStyle = "rgb(255, 255, 231)";
		context.fillRect(a.x - 0.5, a.y - 0.5, a.width, a.height);
		context.strokeStyle = "#000";
		context.lineWidth = 1;
		context.strokeRect(a.x - 0.5, a.y - 0.5, a.width, a.height);
		context.fillStyle = "#000";
		context.fillText(text, b.x, b.y + 13);
	}
};
