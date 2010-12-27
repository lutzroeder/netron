
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

Connector.prototype.isValid = function(value)
{
	if (value === this)
	{
		return false;
	}
	var t1 = this.template.type.split(' ');
	if (!t1.contains("[array]") && (this.connections.length == 1))
	{
		return false;
	}
	if (value instanceof Connector)
	{	
		var t2 = value.template.type.split(' ');
		if ((t1[0] != t2[0]) ||
		(this.owner == value.owner) || 
			(t1.contains("[in]") && !t2.contains("[out]")) || 
			(t1.contains("[out]") && !t2.contains("[in]")) || 
			(!t2.contains("[array]") && (value.connections.length == 1)))
		{
			return false;
		}
	}
	return true;
};

Connector.prototype.paint = function(context, other)
{
	var rectangle = this.getRectangle();
	var strokeStyle = this.owner.owner.theme.connectorBorder; 
	var fillStyle = this.owner.owner.theme.connector;
	if (this.hover)
	{
		strokeStyle = this.owner.owner.theme.connectorHoverBorder; 
		fillStyle = this.owner.owner.theme.connectorHover;
		if (!this.isValid(other))
		{
			fillStyle = "#f00";			
		}
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
