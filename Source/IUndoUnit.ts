module Netron
{
	export interface IUndoUnit
	{
		undo(): void;
		redo(): void;
		isEmpty: bool;
	}
}
