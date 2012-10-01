module Netron
{
	export class Connection implements ISelectable
	{
		private _from: Connector;
		private _to: Connector;
		private _toPoint: Point = null;
		private _selected: bool;
		private _hover: bool;

		constructor(from: Connector, to: Connector)
		{
			this._from = from;
			this._to = to;
		}

		public get from(): Connector
		{
			return this._from;
		}

		public get to(): Connector
		{
			return this._to;
		}

		public get selected(): bool
		{
			return this._selected;
		}

		public set selected(value: bool)
		{
			this._selected = value;
			this.invalidate();
		}

		public get hover(): bool
		{
			return this._hover;
		}

		public set hover(value: bool)
		{
			this._hover = value;
		}

		public updateToPoint(toPoint: Point)
		{
			this._toPoint = toPoint;
		}

		public remove()
		{
			this.invalidate();
			if ((this._from !== null) && (this._from.connections.contains(this)))
			{
				this._from.connections.remove(this);
			}
			if ((this._to !== null) && (this._to.connections.contains(this)))
			{
				this._to.connections.remove(this);
			}
			this._from = null;
			this._to = null;
		}

		public insert(from: Connector, to: Connector)
		{
			this._from = from;
			this._to = to;
			this._from.connections.push(this);
			this._from.invalidate();
			this._to.connections.push(this);
			this._to.invalidate();
			this.invalidate();
		}

		public getCursor(point: Point): string
		{
			return Cursors.select;
		}

		public hitTest(rectangle: Rectangle): bool
		{
			if ((this.from !== null) && (this.to !== null))
			{
				var p1: Point = this.from.element.getConnectorPosition(this.from);
				var p2: Point = this.to.element.getConnectorPosition(this.to);
				if ((rectangle.width !== 0) || (rectangle.height !== 0))
				{
					return (rectangle.contains(p1) && rectangle.contains(p2));
				}
				
				var p: Point = rectangle.topLeft;

				// p1 must be the leftmost point
				if (p1.x > p2.x) { var temp = p2; p2 = p1; p1 = temp; }

				var r1: Rectangle = new Rectangle(p1.x, p1.y, 0, 0);
				var r2: Rectangle = new Rectangle(p2.x, p2.y, 0, 0);
				r1.inflate(3, 3);
				r2.inflate(3, 3);

				if (r1.union(r2).contains(p))
				{
					if ((p1.x == p2.x) || (p1.y == p2.y)) // straight line
					{
						return true;
					}
					else if (p1.y < p2.y)
					{
						var o1 = r1.x + (((r2.x - r1.x) * (p.y - (r1.y + r1.height))) / ((r2.y + r2.height) - (r1.y + r1.height)));
						var u1 = (r1.x + r1.width) + ((((r2.x + r2.width) - (r1.x + r1.width)) * (p.y - r1.y)) / (r2.y - r1.y));
						return ((p.x > o1) && (p.x < u1));
					}
					else
					{
						var o2 = r1.x + (((r2.x - r1.x) * (p.y - r1.y)) / (r2.y - r1.y));
						var u2 = (r1.x + r1.width) + ((((r2.x + r2.width) - (r1.x + r1.width)) * (p.y - (r1.y + r1.height))) / ((r2.y + r2.height) - (r1.y + r1.height)));
						return ((p.x > o2) && (p.x < u2));
					}
				}
			}
			return false;
		}

		public invalidate()
		{
			if (this._from !== null)
			{
				this._from.invalidate();
			}
			if (this._to !== null)
			{
				this._to.invalidate();
			}
		}

		public paint(context: CanvasRenderingContext2D)
		{
			context.strokeStyle = this._from.element.graph.theme.connection;
			context.lineWidth = (this._hover) ? 2 : 1;
			this.paintLine(context, this._selected);
		}

		public paintTrack(context: CanvasRenderingContext2D)
		{
			context.strokeStyle = this.from.element.graph.theme.connection;
			context.lineWidth = 1;
			this.paintLine(context, true);
		}

		public paintLine(context: CanvasRenderingContext2D, dashed: bool)
		{
			if (this._from !== null)
			{
				var start: Point = this._from.element.getConnectorPosition(this.from);
				var end: Point = (this._to !== null) ? this._to.element.getConnectorPosition(this.to) : this._toPoint;
				if ((start.x != end.x) || (start.y != end.y))
				{
					context.beginPath();
					if (dashed)
					{
						LineHelper.dashedLine(context, start.x, start.y, end.x, end.y);
					}
					else
					{
						context.moveTo(start.x - 0.5, start.y - 0.5);
						context.lineTo(end.x - 0.5, end.y - 0.5);
					}
					context.closePath();
					context.stroke();
				}
			}
		}
	}
}