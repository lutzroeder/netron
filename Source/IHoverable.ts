module Netron
{
	export interface IHoverable
	{
		hover: bool;
		getCursor(point: Point): string;
	}
}