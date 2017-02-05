module Netron
{   
    export class LineHelper
    {
        static dashedLine(context: CanvasRenderingContext2D, x1: number, y1: number, x2: number, y2: number)
        {
            context.moveTo(x1, y1);

            var dx: number = x2 - x1;
            var dy: number = y2 - y1;
            var count: number = Math.floor(Math.sqrt(dx * dx + dy * dy) / 3); // dash length
            var ex: number = dx / count;
            var ey: number = dy / count;

            var q: number = 0;
            while (q++ < count) 
            {
                x1 += ex;
                y1 += ey;
                if (q % 2 === 0)
                { 
                    context.moveTo(x1, y1);
                }
                else
                {
                    context.lineTo(x1, y1);
                }
            }
            if (q % 2 === 0)
            {
                context.moveTo(x2, y2);
            }
            else
            {
                context.lineTo(x2, y2);
            }
        }
    }
}