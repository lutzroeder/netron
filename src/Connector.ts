module Netron
{
    export class Connector implements IHoverable
    {
        private _element: Element;
        private _template: IConnectorTemplate;
        private _connections: Connection[] = [];
        private _hover: boolean = false;

        constructor(element: Element, template: IConnectorTemplate)
        {
            this._element = element;
            this._template = template;
        }

        private getRectangle() : Rectangle
        {
            var point: Point = this._element.getConnectorPosition(this);
            var rectangle: Rectangle = new Rectangle(point.x, point.y, 0, 0);
            rectangle.inflate(3, 3);
            return rectangle;
        }

        public get element(): Element
        {
            return this._element;
        }

        public get template(): IConnectorTemplate
        {
            return this._template;
        }

        public get connections(): Connection[]
        {
            return this._connections;
        }

        public get hover(): boolean
        {
            return this._hover;
        }

        public set hover(value: boolean)
        {   
            this._hover = value;
        }

        public getCursor(point: Point): string
        {
            return Cursors.grip;
        }

        public hitTest(rectangle: Rectangle): boolean
        {
            if ((rectangle.width === 0) && (rectangle.height === 0))
            {
                return this.getRectangle().contains(rectangle.topLeft);
            }
            return rectangle.contains(this.getRectangle().topLeft);
        }

        public invalidate()
        {
        }

        public isAssignable(connector: Connector): boolean
        {
            if (connector === this)
            {
                return false;
            }

            var t1: string[] = this._template.type.split(' ');
            if (!t1.contains("[array]") && (this._connections.length == 1))
            {
                return false;
            }

            if (connector instanceof Connector)
            {   
                var t2: string[] = connector._template.type.split(' ');
                if ((t1[0] != t2[0]) ||
                    (this._element == connector.element) || 
                    (t1.contains("[in]") && !t2.contains("[out]")) || 
                    (t1.contains("[out]") && !t2.contains("[in]")) || 
                    (!t2.contains("[array]") && (connector.connections.length == 1)))
                {
                    return false;
                }
            }

            return true;
        }

        public paint(context: CanvasRenderingContext2D, other)
        {
            var rectangle: Rectangle = this.getRectangle();
            var strokeStyle: string = this._element.graph.theme.connectorBorder; 
            var fillStyle: string = this._element.graph.theme.connector;
            if (this._hover)
            {
                strokeStyle = this._element.graph.theme.connectorHoverBorder; 
                fillStyle = this._element.graph.theme.connectorHover;
                if (!this.isAssignable(other))
                {
                    fillStyle = "#f00";         
                }
            }

            context.lineWidth = 1;
            context.strokeStyle = strokeStyle;
            context.lineCap = "butt";
            context.fillStyle = fillStyle;
            context.fillRect(rectangle.x - 0.5, rectangle.y - 0.5, rectangle.width, rectangle.height);
            context.strokeRect(rectangle.x - 0.5, rectangle.y - 0.5, rectangle.width, rectangle.height);

            if (this._hover)
            {
                // Tooltip
                var text = ("description" in this._template) ? this._template.description : this._template.name;
                context.textBaseline = "bottom";
                context.font = "8.25pt Tahoma";
                var size: TextMetrics = context.measureText(text);
                var sizeHeight = 14;
                var sizeWidth = size.width;
                var a: Rectangle = new Rectangle(rectangle.x - Math.floor(size.width / 2), rectangle.y + sizeHeight + 6, sizeWidth, sizeHeight);
                var b: Rectangle = new Rectangle(a.x, a.y, a.width, a.height);
                a.inflate(4, 1);
                context.fillStyle = "rgb(255, 255, 231)";
                context.fillRect(a.x - 0.5, a.y - 0.5, a.width, a.height);
                context.strokeStyle = "#000";
                context.lineWidth = 1;
                context.strokeRect(a.x - 0.5, a.y - 0.5, a.width, a.height);
                context.fillStyle = "#000";
                context.fillText(text, b.x, b.y + 13);
            }
        }
    }
}