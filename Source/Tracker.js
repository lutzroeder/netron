
Tracker = function(rectangle, resizable)
{
	this.rectangle = new Rectangle(rectangle.x, rectangle.y, rectangle.width, rectangle.height);
	this.resizable = resizable;
	this.track = false;
}

Tracker.prototype.hitTest = function(point)
{
	// (0, 0) element, (-1, -1) top-left, (+1, +1) bottom-right
	if (this.resizable)
	{
		for (var x = -1; x <= +1; x++)
		{
			for (var y = -1; y <= +1; y++)
			{
				if ((x !== 0) || (y !== 0))
				{
					var hit = new Point(x, y);
					if (this.getGripRectangle(hit).contains(point))
					{
						return hit;
					}
				}
			}
		}
	}

	if (this.rectangle.contains(point))
	{
		return new Point(0, 0);
	}

	return new Point(-2, -2);
};

Tracker.prototype.getGripRectangle = function(point)
{
	var r = new Rectangle(0, 0, 7, 7);
	if (point.x <   0) { r.x = this.rectangle.x - 7; }
	if (point.x === 0) { r.x = this.rectangle.x + Math.floor(this.rectangle.width / 2) - 3; }
	if (point.x >   0) { r.x = this.rectangle.x + this.rectangle.width + 1; }
	if (point.y <   0) { r.y = this.rectangle.y - 7; }
	if (point.y === 0) { r.y = this.rectangle.y + Math.floor(this.rectangle.height / 2) - 3; }
	if (point.y >   0) { r.y = this.rectangle.y + this.rectangle.height + 1; }
	return r;
};

Tracker.prototype.getCursor = function(point)
{
	var hit = this.hitTest(point);
	if ((hit.x === 0) && (hit.y === 0))
	{
		return (this.track) ? Cursors.move : Cursors.select;
	}
	if ((hit.x >= -1) && (hit.x <= +1) && (hit.y >= -1) && (hit.y <= +1) && this.resizable) 
	{
		if (hit.x === -1 && hit.y === -1) { return "nw-resize"; }
		if (hit.x === +1 && hit.y === +1) { return "se-resize"; }
		if (hit.x === -1 && hit.y === +1) { return "sw-resize"; }
		if (hit.x === +1 && hit.y === -1) { return "ne-resize"; }
		if (hit.x ===  0 && hit.y === -1) { return "n-resize";  }
		if (hit.x ===  0 && hit.y === +1) { return "s-resize";  }
		if (hit.x === +1 && hit.y ===  0) { return "e-resize";  }
		if (hit.x === -1 && hit.y ===  0) { return "w-resize";  }
	}
	return null;
};

Tracker.prototype.start = function(point, handle)
{
	if ((handle.x >= -1) && (handle.x <= +1) && (handle.y >= -1) && (handle.y <= +1))
	{
		this.handle = handle;
		this.currentPoint = point;
		this.track = true;
	}
};

Tracker.prototype.move = function(point)
{
	var h = this.handle;
	var a = new Point(0, 0);
	var b = new Point(0, 0);
	if ((h.x == -1) || ((h.x === 0) && (h.y === 0))) { a.x = point.x - this.currentPoint.x; }
	if ((h.y == -1) || ((h.x === 0) && (h.y === 0))) { a.y = point.y - this.currentPoint.y; }
	if ((h.x == +1) || ((h.x === 0) && (h.y === 0))) { b.x = point.x - this.currentPoint.x; }
	if ((h.y == +1) || ((h.x === 0) && (h.y === 0))) { b.y = point.y - this.currentPoint.y; }
	var tl = new Point(this.rectangle.x, this.rectangle.y);
	var br = new Point(this.rectangle.x + this.rectangle.width, this.rectangle.y + this.rectangle.height);
	tl.x += a.x;
	tl.y += a.y;
	br.x += b.x;
	br.y += b.y;
	this.rectangle.x = tl.x;
	this.rectangle.y = tl.y;
	this.rectangle.width = br.x - tl.x;
	this.rectangle.height = br.y - tl.y;
	this.currentPoint = point;
};

Tracker.prototype.paint = function(context)
{
	if (this.resizable)
	{
		for (var x = -1; x <= +1; x++)
		{
			for (var y = -1; y <= +1; y++)
			{
				if ((x !== 0) || (y !== 0))
				{
					var rectangle = this.getGripRectangle(new Point(x, y));
					context.fillStyle = "#ffffff";
					context.strokeStyle = "#000000";
					context.lineWidth = 1;
					context.fillRect(rectangle.x - 0.5, rectangle.y - 0.5, rectangle.width - 1, rectangle.height - 1);
					context.strokeRect(rectangle.x - 0.5, rectangle.y - 0.5, rectangle.width - 1, rectangle.height - 1);
				}
			}
		}
	}
};
