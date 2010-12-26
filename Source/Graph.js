
function Graph(element)
{
	this.canvas = element;
	this.canvas.style.background = "#fff";
	this.canvas.focus();
	this.context = this.canvas.getContext("2d");
	this.mousePosition = new Point(0, 0);
	this.undoService = new UndoService();
	this.elements = [];
	this.activeTemplate = null;
	this.activeObject = null;
	this.newElement = null;
	this.newConnection = null;
	this.selection = null;
	this.track = false;

	this.mouseDownHandler = this.mouseDown.delegate(this);
	this.mouseUpHandler = this.mouseUp.delegate(this);
	this.mouseMoveHandler = this.mouseMove.delegate(this);
	this.doubleClickHandler = this.doubleClick.delegate(this);
	this.keyDownHandler = this.keyDown.delegate(this);
	this.keyUpHandler = this.keyUp.delegate(this);

	this.canvas.addEventListener("mousedown", this.mouseDownHandler, false);
	this.canvas.addEventListener("mouseup", this.mouseUpHandler, false);
	this.canvas.addEventListener("mousemove", this.mouseMoveHandler, false);
	this.canvas.addEventListener("dblclick", this.doubleClickHandler, false);
	this.canvas.addEventListener("keydown", this.keyDownHandler, false);
	this.canvas.addEventListener("keyup", this.keyUpHandler, false);
}

Graph.prototype.dispose = function()
{
	if (this.canvas !== null)
	{
		this.canvas.removeEventListener("mousedown", this.mouseDownHandler);
		this.canvas.removeEventListener("mouseup", this.mouseUpHandler);
		this.canvas.removeEventListener("mousemove", this.mouseMoveHandler);
		this.canvas.removeEventListener("dblclick", this.doubleClickHandler);
		this.canvas.removeEventListener("keydown", this.keyDownHandler);
		this.canvas.removeEventListener("keyup", this.keyUpHandler);	
		this.canvas = null;
		this.context = null;
	}
};

Graph.prototype.mouseDown = function(e)
{
	e.preventDefault();
	this.canvas.focus();
	this.updateMousePosition(e);
	var point = this.mousePosition;

	if (e.button === 0) // left-click
	{
		// alt+click allows fast creation of element using the active template
		if ((this.newElement === null) && (e.altKey))
		{
			this.createElement(this.activeTemplate);
		}

		if (this.newElement !== null)
		{
			this.undoService.begin();
			this.newElement.invalidate();
			this.newElement.rectangle = new Rectangle(point.x, point.y, this.newElement.rectangle.width, this.newElement.rectangle.height);
			this.newElement.invalidate();
			this.undoService.add(new InsertElementUndoUnit(this.newElement, this));
			this.undoService.commit();
			this.newElement = null;
		}
		else
		{
			this.selection = null;
			this.updateActiveObject(point);
			if (this.activeObject === null)
			{
				// start selection
				this.selection = new Selection(point);			
			}
			else
			{
				// start connection
				if (this.activeObject instanceof Connector)
				{
					this.newConnection = new Connection(this.activeObject, null);
					this.newConnection.toPoint = point;
					this.activeObject.invalidate();
				}
				else
				{
					// select object
					if (!this.activeObject.selected)
					{
						this.undoService.begin();
						var selectionUndoUnit = new SelectionUndoUnit();
						if (!e.shiftKey)
						{
							this.deselectAll(selectionUndoUnit);
						}
						selectionUndoUnit.select(this.activeObject);
						this.undoService.add(selectionUndoUnit);
						this.undoService.commit();
					}

					// start tracking
					var hit = new Point(0, 0);
					if (this.activeObject instanceof Element)
					{
						hit = this.activeObject.tracker.hitTest(point);
					}
					for (var i = 0; i < this.elements.length; i++)
					{
						var element = this.elements[i];
						if (element.tracker !== null)
						{
							element.tracker.start(point, hit);
						}
					}

					this.track = true;
				}
			}
		}
	}
	else if (e.button == 2) // right-click
	{
		if ((this.activeObject !== null) && (!this.activeObject.selected))
		{
			this.undoService.begin();
			var deselectUndoUnit = new SelectionUndoUnit();
			this.deselectAll(deselectUndoUnit);
			this.undoService.add(deselectUndoUnit);
			this.undoService.commit();
		}
	}

	this.update();
	this.updateMouseCursor();
};

Graph.prototype.mouseUp = function(e)
{
	e.preventDefault();
	this.updateMousePosition(e);
	var point = this.mousePosition;
	
	if (e.button === 0) // left-click
	{
		if (this.newConnection !== null)
		{
			this.updateActiveObject(point);
			this.newConnection.invalidate();
			if ((this.activeObject !== null) && (this.activeObject instanceof Connector))
			{
				if (this.activeObject != this.newConnection.from)
				{
					this.undoService.begin();
					this.undoService.add(new InsertConnectionUndoUnit(this.newConnection, this.newConnection.from, this.activeObject));
					this.undoService.commit();
				}
			}

			this.newConnection = null;
		}

		if (this.selection !== null)
		{
			this.undoService.begin();
			var selectionUndoUnit = new SelectionUndoUnit();

			var rectangle = this.selection.getRectangle();
			if ((this.activeObject === null) || (!this.activeObject.selected))
			{
				if (!e.shiftKey)
				{
					this.deselectAll(selectionUndoUnit);
				}
			}

			if ((rectangle.width !== 0) || (rectangle.weight !== 0))
			{
				this.selectAll(selectionUndoUnit, rectangle);
			}

			this.undoService.add(selectionUndoUnit);
			this.undoService.commit();
			this.selection = null;
		}

		if (this.track)
		{
			this.undoService.begin();
			for (var i = 0; i < this.elements.length; i++)
			{
				var element = this.elements[i];
				if (element.tracker !== null)
				{
					element.tracker.track = false;
					element.invalidate();
					var r1 = element.getRectangle();
					var r2 = element.tracker.rectangle;
					if ((r1.x != r2.x) || (r1.y != r2.y) || (r1.width != r2.width) || (r1.height != r2.height))
					{
						this.undoService.add(new TransformUndoUnit(element, r1, r2));
					}
				}
			}

			this.undoService.commit();
			this.track = false;
			this.updateActiveObject(point);
		}
	}

	this.update();
	this.updateMouseCursor();
};

Graph.prototype.mouseMove = function(e)
{
	e.preventDefault(); 
	this.updateMousePosition(e);
	var point = this.mousePosition;

	if (this.newElement !== null)
	{
		// placing new element
		this.newElement.invalidate();
		this.newElement.rectangle = new Rectangle(point.x, point.y, this.newElement.rectangle.width, this.newElement.rectangle.height);
		this.newElement.invalidate();
	}

	if (this.track)
	{
		// moving selected elements
		for (var i = 0; i < this.elements.length; i++)
		{
			var element = this.elements[i];
			if (element.tracker !== null)
			{
				element.invalidate();
				element.tracker.move(point);
				element.invalidate();
			}
		}
	}

	if (this.newConnection !== null)
	{
		// connecting two connectors
		this.newConnection.invalidate();
		this.newConnection.toPoint = point;
		this.newConnection.invalidate();
	}

	if (this.selection !== null)
	{
		this.selection.currentPoint = point;
	}

	this.updateActiveObject(point);
	this.update();
	this.updateMouseCursor();
};

Graph.prototype.doubleClick = function(e)
{
	e.preventDefault();
	this.updateMousePosition(e);
	var point = this.mousePosition;

	if (e.button === 0) // left-click
	{
		this.updateActiveObject(point);
		if ((this.activeObject !== null) && (this.activeObject instanceof Element) && (this.activeObject.template !== null) && ("edit" in this.activeObject.template))
		{
			this.activeObject.template.edit(this.activeObject, point);
			this.update();
		}
	}
};

Graph.prototype.keyDown = function(e)
{
	if ((e.ctrlKey || e.metaKey) && !e.altKey) // ctrl or option
	{
		if (e.keyCode == 65) // A - select all
		{
			this.undoService.begin();
			var selectionUndoUnit = new SelectionUndoUnit();
			this.selectAll(selectionUndoUnit, null);
			this.undoService.add(selectionUndoUnit);
			this.undoService.commit();
			this.update();
			this.updateActiveObject(this.mousePosition);
			this.updateMouseCursor();
			e.preventDefault();
		}

		if ((e.keyCode == 90) && (!e.shiftKey)) // Z - undo
		{
			this.undoService.undo();
			this.update();
			this.updateActiveObject(this.mousePosition);
			this.updateMouseCursor();
			e.preventDefault();
		}
		
		if (((e.keyCode == 90) && (e.shiftKey)) || (e.keyCode == 89)) // Y - redo
		{
			this.undoService.redo();
			this.update();
			this.updateActiveObject(this.mousePosition);
			this.updateMouseCursor();
			e.preventDefault();
		}
	}

	if ((e.keyCode == 46) || (e.keyCode == 8)) // DEL - delete
	{
		this.deleteSelection();
		this.update();
		this.updateActiveObject(this.mousePosition);
		this.updateMouseCursor();
		e.preventDefault();
	}

	if (e.keyCode == 27) // ESC
	{
		this.newElement = null;
		this.newConnection = null;

		this.track = false;
		for (var i = 0; i < this.elements.length; i++)
		{
			var element = this.elements[i];
			if (element.tracker !== null)
			{
				element.tracker.track = false;
			}
		}
		
		this.update();
		this.updateActiveObject(this.mousePosition);
		this.updateMouseCursor();
		e.preventDefault();
	}
};

Graph.prototype.keyUp = function(e)
{
	this.updateMouseCursor();
};

Graph.prototype.deleteSelection = function()
{
	this.undoService.begin();
	
	for (var i = 0; i < this.elements.length; i++)
	{
		var element = this.elements[i];
		if (element.selected)
		{
			this.undoService.add(new DeleteElementUndoUnit(element));
		}

		for (var j = 0; j < element.connectors.length; j++)
		{
			var connector = element.connectors[j];
			for (var k = 0; k < connector.connections.length; k++)
			{
				var connection = connector.connections[k];
				if (element.selected || connection.selected)
				{
					this.undoService.add(new DeleteConnectionUndoUnit(connection));
				}
			}
		}
	}
	
	this.undoService.commit();
};

Graph.prototype.selectAll = function(selectionUndoUnit, rectangle)
{
	for (var i = 0; i < this.elements.length; i++)
	{
		var element = this.elements[i];
		if ((rectangle === null) || (element.hitTest(rectangle)))
		{
			selectionUndoUnit.select(element);
		}

		for (var j = 0; j < element.connectors.length; j++)
		{
			var connector = element.connectors[j];
			for (var k = 0; k < connector.connections.length; k++)
			{
				var connection = connector.connections[k];
				if ((rectangle === null) || (connection.hitTest(rectangle)))
				{
					selectionUndoUnit.select(connection);
				}
			}
		}
	}
};

Graph.prototype.deselectAll = function(selectionUndoUnit)
{
	for (var i = 0; i < this.elements.length; i++)
	{
		var element = this.elements[i];
		selectionUndoUnit.deselect(element);

		for (var j = 0; j < element.connectors.length; j++)
		{
			var connector = element.connectors[j];
			for (var k = 0; k < connector.connections.length; k++)
			{
				var connection = connector.connections[k];
				selectionUndoUnit.deselect(connection);
			}
		}
	}
};

Graph.prototype.updateActiveObject = function(point)
{
	var hitObject = this.hitTest(point);
	if (hitObject != this.activeObject)
	{
		if (this.activeObject !== null) 
		{
			this.activeObject.hover = false;
		}
		this.activeObject = hitObject;
		if (this.activeObject !== null)
		{
			this.activeObject.hover = true;
		}
	}
};

Graph.prototype.hitTest = function(point)
{
	var i, j, k;
	var element, connector, connection;

	var rectangle = new Rectangle(point.x, point.y, 0, 0);

	for (i = 0; i < this.elements.length; i++)
	{
		element = this.elements[i];
		for (j = 0; j < element.connectors.length; j++)
		{
			connector = element.connectors[j];
			if (connector.hitTest(rectangle))
			{
				return connector;
			}
		}
	}

	for (i = 0; i < this.elements.length; i++)
	{
		element = this.elements[i];
		if (element.hitTest(rectangle))
		{
			return element;
		}
	}

	for (i = 0; i < this.elements.length; i++)
	{
		element = this.elements[i];
		for (j = 0; j < element.connectors.length; j++)
		{
			connector = element.connectors[j];
			for (k = 0; k < connector.connections.length; k++)
			{
				connection = connector.connections[k];
				if (connection.hitTest(rectangle))
				{
					return connection;
				}
			}
		}
	}

	return null;
};

Graph.prototype.updateMouseCursor = function()
{	
	if (this.newConnection !== null)
	{
		this.canvas.style.cursor = ((this.activeObject !== null) && (this.activeObject instanceof Connector)) ? this.activeObject.getCursor(this.mousePosition) : Cursors.cross;
	}
	else
	{
		this.canvas.style.cursor = (this.activeObject !== null) ? this.activeObject.getCursor(this.mousePosition) : Cursors.arrow;
	}
};

Graph.prototype.updateMousePosition = function(e)
{
	this.mousePosition = new Point(e.pageX, e.pageY);
	var node = this.canvas;
	while (node != null)
	{
		this.mousePosition.x -= node.offsetLeft;
		this.mousePosition.y -= node.offsetTop;
		node = node.offsetParent;
	}
	
//	this.mousePosition = new Point(e.pageX - this.canvas.offsetLeft, e.pageY - this.canvas.offsetTop);
};

Graph.prototype.addElement = function(template, point, content)
{
	this.activeTemplate = template;
	var element = new Element(template, point);
	element.content = content;
	element.insertInto(this);
	element.invalidate();
	return element;
};

Graph.prototype.createElement = function(template)
{
	this.activeTemplate = template;
	this.newElement = new Element(template, this.mousePosition);
	this.canvas.focus();
}

Graph.prototype.addConnection = function(connector1, connector2)
{
	var connection = new Connection(connector1, connector2);
	connector1.connections.push(connection);
	connector2.connections.push(connection);
	connector1.invalidate();
	connector2.invalidate();
	connection.invalidate();
	return connection;
};

Graph.prototype.update = function()
{
	var i, j, k;
	var element, connector, connection;
	
	this.context.clearRect(0, 0, this.canvas.width, this.canvas.height);
	
	var connections = [];
	for (i = 0; i < this.elements.length; i++)
	{
		element = this.elements[i];
		for (j = 0; j < element.connectors.length; j++)
		{
			connector = element.connectors[j];
			for (k = 0; k < connector.connections.length; k++)
			{
				connection = connector.connections[k];
				if (!connections.contains(connection))
				{
					connection.paint(this.context);
					connections.push(connection);
				}
			}
		}
	}

	for (i = 0; i < this.elements.length; i++)
	{
		this.context.save();
		this.elements[i].paint(this.context);
		this.context.restore();
	}

	for (i = 0; i < this.elements.length; i++)
	{
		element = this.elements[i];
		for (j = 0; j < element.connectors.length; j++)
		{
			connector = element.connectors[j];

			var hover = false;
			for (k = 0; k < connector.connections.length; k++)
			{
				if (connector.connections[k].hover) { hover = true; }
			}

			if ((element.hover) || (connector.hover) || hover)
			{
				connector.paint(this.context);
			}
			else if (this.newConnection !== null) // TODO check type
			{
				connector.paint(this.context);
			}
		}
	}
	
	if (this.newElement !== null)
	{
		this.newElement.paint(this.context);
	}
	
	if (this.newConnection !== null)
	{
		this.newConnection.paintTrack(this.context);
	}
	
	if (this.selection !== null)
	{
		this.selection.paint(this.context);
	}
};
