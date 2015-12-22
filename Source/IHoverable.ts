module Netron
{
    export interface IHoverable
    {
        hover: boolean;
        getCursor(point: Point): string;
    }
}