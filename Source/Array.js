
Array.prototype.remove = function(obj)
{
	var i = this.length;
	while (i--)
	{
		if (this[i] == obj)
		{
			this.splice(i, 1);
		}
	}
};

Array.prototype.contains = function(obj)
{
	var i = this.length;
	while (i--)
	{
		if (this[i] == obj)
		{
			return true;
		}
	}
	return false;
};
