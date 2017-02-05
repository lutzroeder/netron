module Netron
{
    export class Tracker
    {
        private _rectangle: Rectangle;
        private _resizable: boolean;
        private _track: boolean = false;
        private _handle: Point;
        private _currentPoint: Point;

        constructor(rectangle: Rectangle, resizable: boolean)
        {
            this._rectangle = rectangle.clone();
            this._resizable = resizable;
        }

        public get rectangle(): Rectangle
        {
            return this._rectangle;
        }

        public hitTest(point: Point): Point
        {
            // (0, 0) element, (-1, -1) top-left, (+1, +1) bottom-right
            if (this._resizable)
            {
                for (var x = -1; x <= +1; x++)
                {
                    for (var y = -1; y <= +1; y++)
                    {
                        if ((x !== 0) || (y !== 0))
                        {
                            var hit = new Point(x, y);
                            if (this.getGripRectangle(hit).contains(point))
                            {
                                return hit;
                            }
                        }
                    }
                }
            }

            if (this._rectangle.contains(point))
            {
                return new Point(0, 0);
            }

            return new Point(-2, -2);
        }

        public getGripRectangle(point: Point): Rectangle
        {
            var r: Rectangle = new Rectangle(0, 0, 7, 7);
            if (point.x <   0) { r.x = this._rectangle.x - 7; }
            if (point.x === 0) { r.x = this._rectangle.x + Math.floor(this._rectangle.width / 2) - 3; }
            if (point.x >   0) { r.x = this._rectangle.x + this._rectangle.width + 1; }
            if (point.y <   0) { r.y = this._rectangle.y - 7; }
            if (point.y === 0) { r.y = this._rectangle.y + Math.floor(this._rectangle.height / 2) - 3; }
            if (point.y >   0) { r.y = this._rectangle.y + this._rectangle.height + 1; }
            return r;
        }

        public getCursor(point: Point): string
        {
            var hit: Point = this.hitTest(point);
            if ((hit.x === 0) && (hit.y === 0))
            {
                return (this._track) ? Cursors.move : Cursors.select;
            }
            if ((hit.x >= -1) && (hit.x <= +1) && (hit.y >= -1) && (hit.y <= +1) && this._resizable) 
            {
                if (hit.x === -1 && hit.y === -1) { return "nw-resize"; }
                if (hit.x === +1 && hit.y === +1) { return "se-resize"; }
                if (hit.x === -1 && hit.y === +1) { return "sw-resize"; }
                if (hit.x === +1 && hit.y === -1) { return "ne-resize"; }
                if (hit.x ===  0 && hit.y === -1) { return "n-resize";  }
                if (hit.x ===  0 && hit.y === +1) { return "s-resize";  }
                if (hit.x === +1 && hit.y ===  0) { return "e-resize";  }
                if (hit.x === -1 && hit.y ===  0) { return "w-resize";  }
            }
            return null;
        }

        public start(point: Point, handle: Point)
        {
            if ((handle.x >= -1) && (handle.x <= +1) && (handle.y >= -1) && (handle.y <= +1))
            {
                this._handle = handle;
                this._currentPoint = point;
                this._track = true;
            }
        }

        public stop()
        {
            this._track = false;
        }

        public get track(): boolean
        {
            return this._track;
        }

        public move(point: Point)
        {
            var h: Point = this._handle;
            var a: Point = new Point(0, 0);
            var b: Point = new Point(0, 0);
            if ((h.x == -1) || ((h.x === 0) && (h.y === 0))) { a.x = point.x - this._currentPoint.x; }
            if ((h.y == -1) || ((h.x === 0) && (h.y === 0))) { a.y = point.y - this._currentPoint.y; }
            if ((h.x == +1) || ((h.x === 0) && (h.y === 0))) { b.x = point.x - this._currentPoint.x; }
            if ((h.y == +1) || ((h.x === 0) && (h.y === 0))) { b.y = point.y - this._currentPoint.y; }
            var tl: Point = new Point(this._rectangle.x, this._rectangle.y);
            var br: Point = new Point(this._rectangle.x + this._rectangle.width, this._rectangle.y + this._rectangle.height);
            tl.x += a.x;
            tl.y += a.y;
            br.x += b.x;
            br.y += b.y;
            this._rectangle.x = tl.x;
            this._rectangle.y = tl.y;
            this._rectangle.width = br.x - tl.x;
            this._rectangle.height = br.y - tl.y;
            this._currentPoint = point;
        }

        public updateRectangle(rectangle: Rectangle)
        {
            this._rectangle = rectangle.clone();
        }

        public paint(context: CanvasRenderingContext2D)
        {
            if (this._resizable)
            {
                for (var x: number = -1; x <= +1; x++)
                {
                    for (var y: number = -1; y <= +1; y++)
                    {
                        if ((x !== 0) || (y !== 0))
                        {
                            var rectangle: Rectangle = this.getGripRectangle(new Point(x, y));
                            context.fillStyle = "#ffffff";
                            context.strokeStyle = "#000000";
                            context.lineWidth = 1;
                            context.fillRect(rectangle.x - 0.5, rectangle.y - 0.5, rectangle.width - 1, rectangle.height - 1);
                            context.strokeRect(rectangle.x - 0.5, rectangle.y - 0.5, rectangle.width - 1, rectangle.height - 1);
                        }
                    }
                }
            }
        }
    }
}