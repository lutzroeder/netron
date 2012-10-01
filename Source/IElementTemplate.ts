module Netron
{
	export interface IElementTemplate
	{
		resizable: bool;
		defaultWidth: number;
		defaultHeight: number;
		defaultContent: any;
		connectorTemplates: IConnectorTemplate[];

		paint(element: Element, context: CanvasRenderingContext2D);
		edit(element: Element, context: CanvasRenderingContext2D, point: Point);
	}	
}