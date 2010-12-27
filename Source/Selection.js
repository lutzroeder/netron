
function Selection(startPoint)
{
	this.startPoint = startPoint;
	this.currentPoint = startPoint;
}

Selection.prototype.paint = function(context)
{
	var r = this.getRectangle();
	context.lineWidth = 1;
	context.beginPath();
	context.dashedLine(r.x - 0.5,           r.y - 0.5,            r.x - 0.5 + r.width, r.y - 0.5);
	context.dashedLine(r.x - 0.5 + r.width, r.y - 0.5,            r.x - 0.5 + r.width, r.y - 0.5 + r.height);
	context.dashedLine(r.x - 0.5 + r.width, r.y - 0.5 + r.height, r.x - 0.5,           r.y - 0.5 + r.height);
	context.dashedLine(r.x - 0.5,           r.y - 0.5 + r.height, r.x - 0.5,           r.y - 0.5);
	context.closePath();
	context.stroke();
};

Selection.prototype.getRectangle = function()
{
	var r = new Rectangle(
		(this.startPoint.x <= this.currentPoint.x) ? this.startPoint.x : this.currentPoint.x,
		(this.startPoint.y <= this.currentPoint.y) ? this.startPoint.y : this.currentPoint.y,
		this.currentPoint.x - this.startPoint.x,
		this.currentPoint.y - this.startPoint.y);
	if (r.width < 0) 
	{
		r.width *= -1;
	}
	if (r.height < 0) 
	{
		r.height *= -1;
	}
	return r;
};
