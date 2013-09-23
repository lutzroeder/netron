module Netron
{
	export class Element implements ISelectable
	{
		private _template: IElementTemplate;
		private _rectangle: Rectangle;
		private _graph: Graph = null;
		private _content;
		private _hover: boolean = false;
		private _selected: boolean = false;
		private _tracker: Tracker = null;
		private _connectors: Connector[] = [];

		constructor(template: IElementTemplate, point: Point)
		{
			this._template = template;
			this._content = template.defaultContent;
			this._rectangle = new Rectangle(point.x, point.y, template.defaultWidth, template.defaultHeight);

			for (var i: number = 0; i < template.connectorTemplates.length; i++)
			{
				var connectorTemplate = template.connectorTemplates[i];
				this._connectors.push(new Connector(this, connectorTemplate));
			}	
		}

		public get rectangle(): Rectangle
		{
			return ((this._tracker !== null) && (this._tracker.track)) ? this._tracker.rectangle : this._rectangle;
		}

		public set rectangle(value: Rectangle)
		{
			this.invalidate();
			this._rectangle = value;
			if (this._tracker !== null)
			{
				this._tracker.updateRectangle(value);
			}
			this.invalidate();
		}

		public get template(): IElementTemplate
		{
			return this._template;
		}

		public get graph(): Graph
		{
			return this._graph;
		}

		public get connectors(): Connector[]
		{
			return this._connectors;
		}

		public get tracker(): Tracker
		{
			return this._tracker;
		}

		public get selected(): boolean
		{
			return this._selected;
		}

		public set selected(value: boolean)
		{
			this._selected = value;

			if (this._selected)
			{
				this._tracker = new Tracker(this._rectangle, ("resizable" in this._template) ? this._template.resizable : false);
				this.invalidate();			
			}
			else
			{
				this.invalidate();
				this._tracker = null;
			}
		}

		public get hover(): boolean
		{
			return this._hover;
		}

		public set hover(value: boolean)
		{
			this._hover = value;
		}

		public paint(context: CanvasRenderingContext2D)
		{
			this._template.paint(this, context);
			
			if (this._selected)
			{
				this._tracker.paint(context);
			}
		}

		public invalidate()
		{
		}

		public insertInto(graph: Graph)
		{
			this._graph = graph;
			this._graph.elements.push(this);
		}

		public remove()
		{
			this.invalidate();

			for (var i = 0; i < this._connectors.length; i++)
			{
				var connections = this._connectors[i].connections;
				for (var j = 0; j < connections.length; j++)
				{
					connections[j].remove();
				}
			}
			
			if ((this._graph !== null) && (this._graph.elements.contains(this)))
			{
				this._graph.elements.remove(this);
			}

			this._graph = null;
		}

		public hitTest(rectangle: Rectangle): boolean
		{
			if ((rectangle.width === 0) && (rectangle.height === 0))
			{
				if (this._rectangle.contains(rectangle.topLeft))
				{
					return true;
				}

				if ((this._tracker !== null) && (this._tracker.track))
				{
					var h = this._tracker.hitTest(rectangle.topLeft);
					if ((h.x >= -1) && (h.x <= +1) && (h.y >= -1) && (h.y <= +1))
					{
						return true;
					}
				}

				for (var i: number = 0; i < this._connectors.length; i++)
				{
					if (this._connectors[i].hitTest(rectangle))
					{
						return true;
					}
				}

				return false;
			}

			return rectangle.contains(this._rectangle.topLeft);
		}

		public getCursor(point: Point): string
		{
			if (this._tracker !== null)
			{
				var cursor = this._tracker.getCursor(point);
				if (cursor !== null)
				{
					return cursor;
				}
			}

			if (window.event.shiftKey)
			{
				return Cursors.add;
			}

			return Cursors.select;
		}

		public getConnector(name: string): Connector
		{
			for (var i: number = 0; i < this._connectors.length; i++)
			{
				var connector = this._connectors[i];
				if (connector.template.name == name)
				{
					return connector;
				}
			}
			return null;
		}

		public getConnectorPosition(connector: Connector): Point
		{
			var rectangle: Rectangle = this.rectangle;
			var point: Point = connector.template.getConnectorPosition(this);
			point.x += rectangle.x;
			point.y += rectangle.y;
			return point;
		}

		public setContent(content: any)
		{
			this._graph.setElementContent(this, content);
		}

		public get content(): any
		{
			return this._content;
		}

		public set content(value: any)
		{
			this._content = value;
		}
	}
}