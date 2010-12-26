
CanvasRenderingContext2D.prototype.dashedLine = function(x1, y1, x2, y2)
{
	this.moveTo(x1, y1);
	var dx = x2 - x1;
	var dy = y2 - y1;
	var count = Math.floor(Math.sqrt(dx * dx + dy * dy) / 3); // dash length
	var ex = dx / count;
	var ey = dy / count;

	var q = 0;
	while (q++ < count) 
	{
		x1 += ex;
		y1 += ey;
		if (q % 2 === 0)
		{ 
			this.moveTo(x1, y1);
		}
		else
		{
			this.lineTo(x1, y1);
		}
	}
	if (q % 2 === 0)
	{
		this.moveTo(x2, y2);
	}
	else
	{
		this.lineTo(x2, y2);
	}
};

CanvasRenderingContext2D.prototype.roundedRect = function(x, y, w, h, radius)
{
  context.beginPath();
  context.moveTo(x + radius, y);
  context.lineTo(x + width - radius, y);
  context.quadraticCurveTo(x + width, y, x + width, y + radius);
  context.lineTo(x + width, y + height - radius);
  context.quadraticCurveTo(x + width, y + height, x + width - radius, y + height);
  context.lineTo(x + radius, y + height);
  context.quadraticCurveTo(x, y + height, x, y + height - radius);
  context.lineTo(x, y + radius);
  context.quadraticCurveTo(x, y, x + radius, y);
  context.closePath();
};
