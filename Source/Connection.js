
function Connection(from, to)
{
	this.from = from;
	this.to = to;
	this.toPoint = null;
}

Connection.prototype.select = function()
{
	this.selected = true;
	this.invalidate();
};

Connection.prototype.deselect = function()
{
	this.selected = false;
	this.invalidate();
};

Connection.prototype.remove = function()
{
	this.invalidate();
	if ((this.from !== null) && (this.from.connections.contains(this)))
	{
		this.from.connections.pop(this);
	}
	if ((this.to !== null) && (this.to.connections.contains(this)))
	{
		this.to.connections.pop(this);
	}
	this.from = null;
	this.to = null;
};

Connection.prototype.insert = function(from, to)
{
	this.from = from;
	this.to = to;
	this.from.connections.push(this);
	this.from.invalidate();
	this.to.connections.push(this);
	this.to.invalidate();
	this.invalidate();
};

Connection.prototype.getCursor = function(point)
{
	return Cursors.select;
};

Connection.prototype.hitTest = function(rectangle)
{
	if ((this.from !== null) && (this.to !== null))
	{
		var p1 = this.from.owner.getConnectorPosition(this.from);
		var p2 = this.to.owner.getConnectorPosition(this.to);
		if ((rectangle.width !== 0) || (rectangle.height !== 0))
		{
			return (rectangle.contains(p1) && rectangle.contains(p2));
		}
		
		var p = rectangle.topLeft();

		// p1 must be the leftmost point
		if (p1.x > p2.x) { var temp = p2; p2 = p1; p1 = temp; }

		var r1 = new Rectangle(p1.x, p1.y, 0, 0);
		var r2 = new Rectangle(p2.x, p2.y, 0, 0);
		r1.inflate(3, 3);
		r2.inflate(3, 3);

		if (r1.union(r2).contains(p))
		{
			if (p1.y < p2.y)
			{
				var o1 = r1.x + (((r2.x - r1.x) * (p.y - (r1.y + r1.height))) / ((r2.y + r2.height) - (r1.y + r1.height)));
				var u1 = (r1.x + r1.width) + ((((r2.x + r2.width) - (r1.x + r1.width)) * (p.y - r1.y)) / (r2.y - r1.y));
				return ((p.x > o1) && (p.x < u1));
			}
			else
			{
				var o2 = r1.x + (((r2.x - r1.x) * (p.y - r1.y)) / (r2.y - r1.y));
				var u2 = (r1.x + r1.width) + ((((r2.x + r2.width) - (r1.x + r1.width)) * (p.y - (r1.y + r1.height))) / ((r2.y + r2.height) - (r1.y + r1.height)));
				return ((p.x > o2) && (p.x < u2));
			}
		}
	}
	return false;
};

Connection.prototype.invalidate = function()
{
	if (this.from !== null)
	{
		this.from.invalidate();
	}
	if (this.to !== null)
	{
		this.to.invalidate();
	}
};

Connection.prototype.paint = function(context)
{
	context.strokeStyle = "#000000";
	context.lineWidth = (this.hover) ? 2 : 1;
	this.paintLine(context, this.selected);
};

Connection.prototype.paintTrack = function(context)
{
	context.strokeStyle = "#000000";
	context.lineWidth = 1;
	this.paintLine(context, true);
};

Connection.prototype.paintLine = function(context, dashed)
{
	if (this.from !== null)
	{
		var start = this.from.owner.getConnectorPosition(this.from);
		var end = (this.to !== null) ? this.to.owner.getConnectorPosition(this.to) : this.toPoint;
		if ((start.x != end.x) || (start.y != end.y))
		{
			context.beginPath();
			if (dashed)
			{
				context.dashedLine(start.x, start.y, end.x, end.y);
			}
			else
			{
				context.moveTo(start.x - 0.5, start.y - 0.5);
				context.lineTo(end.x - 0.5, end.y - 0.5);
			}
			context.closePath();
			context.stroke();
		}
	}
};
