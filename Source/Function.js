
Function.prototype.bind = function(obj)
{
	var fn = this;
	return function()
	{
		return fn.apply(obj, arguments);
	};
};
