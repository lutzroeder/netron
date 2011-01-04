
Element = function(template, point)
{
	this.template = template;
	this.rectangle = new Rectangle(point.x, point.y, template.defaultWidth, template.defaultHeight);
	this.content = template.defaultContent;
	this.owner = null;
	this.hover = false;
	this.selected = false;
	this.tracker = null;
	this.connectors = [];
	for (var i = 0; i < template.connectorTemplates.length; i++)
	{
		var connectorTemplate = template.connectorTemplates[i];
		this.connectors.push(new Connector(this, connectorTemplate));
	}	
}

Element.prototype.select = function()
{
	this.selected = true;
	this.tracker = new Tracker(this.rectangle, ("resizable" in this.template) ? this.template.resizable : false);
	this.invalidate();
};

Element.prototype.deselect = function()
{
	this.selected = false;
	this.invalidate();
	this.tracker = null;
};

Element.prototype.getRectangle = function()
{
	return ((this.tracker !== null) && (this.tracker.track)) ? this.tracker.rectangle : this.rectangle;
};

Element.prototype.getPageRectangle = function()
{
	var rectangle = this.getRectangle();
	rectangle = new Rectangle(rectangle.x, rectangle.y, rectangle.width, rectangle.height);
	var canvas = this.owner.canvas;
	rectangle.x += canvas.offsetLeft;
	rectangle.y += canvas.offsetTop;
	return rectangle;
};

Element.prototype.setRectangle = function(rectangle)
{
	this.invalidate();
	this.rectangle = rectangle;
	if (this.tracker !== null)
	{
		this.tracker.rectangle = new Rectangle(rectangle.x, rectangle.y, rectangle.width, rectangle.height);
	}
	this.invalidate();
};

Element.prototype.paint = function(context)
{
	this.template.paint(this, context);
	
	if (this.selected)
	{
		this.tracker.paint(context);
	}
};

Element.prototype.invalidate = function()
{
};

Element.prototype.insertInto = function(owner)
{
	this.owner = owner;
	this.owner.elements.push(this);
};

Element.prototype.remove = function()
{
	this.invalidate();

	for (var i = 0; i < this.connectors.length; i++)
	{
		var connections = this.connectors[i].connections;
		for (var j = 0; j < connections.length; j++)
		{
			connections[j].remove();
		}
	}
	
	if ((this.owner !== null) && (this.owner.elements.contains(this)))
	{
		this.owner.elements.remove(this);
	}

	this.owner = null;
};

Element.prototype.hitTest = function(rectangle)
{
	if ((rectangle.width === 0) && (rectangle.height === 0))
	{
		if (this.rectangle.contains(rectangle.topLeft()))
		{
			return true;
		}

		if ((this.tracker !== null) && (this.tracker.track))
		{
			var h = this.tracker.hitTest(rectangle.topLeft());
			if ((h.x >= -1) && (h.x <= +1) && (h.y >= -1) && (h.y <= +1))
			{
				return true;
			}
		}

		for (var i = 0; i < this.connectors.length; i++)
		{
			if (this.connectors[i].hitTest(rectangle))
			{
				return true;
			}
		}

		return false;
	}

	return rectangle.contains(this.rectangle);
};

Element.prototype.getCursor = function(point)
{
	if (this.tracker !== null)
	{
		var cursor = this.tracker.getCursor(point);
		if (cursor !== null)
		{
			return cursor;
		}
	}

	if (window.event.shiftKey)
	{
		return Cursors.add;
	}

	return Cursors.select;
};

Element.prototype.getConnector = function(name)
{
	for (var i = 0; i < this.connectors.length; i++)
	{
		var connector = this.connectors[i];
		if (connector.template.name == name)
		{
			return connector;
		}
	}
	return null;
};

Element.prototype.getConnectorPosition = function(connector)
{
	var rectangle = this.getRectangle();
	var point = connector.template.position(this);
	point.x += rectangle.x;
	point.y += rectangle.y;
	return point;
};

Element.prototype.setContent = function(content)
{
	this.owner.setElementContent(this, content);
};

Element.prototype.getContent = function()
{
	return this.content;
};
