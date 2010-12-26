
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

CanvasRenderingContext2D.prototype.roundedRect = function(x, y, width, height, radius)
{
  this.beginPath();
  this.moveTo(x + radius, y);
  this.lineTo(x + width - radius, y);
  this.quadraticCurveTo(x + width, y, x + width, y + radius);
  this.lineTo(x + width, y + height - radius);
  this.quadraticCurveTo(x + width, y + height, x + width - radius, y + height);
  this.lineTo(x + radius, y + height);
  this.quadraticCurveTo(x, y + height, x, y + height - radius);
  this.lineTo(x, y + radius);
  this.quadraticCurveTo(x, y, x + radius, y);
  this.closePath();
};
