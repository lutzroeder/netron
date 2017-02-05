module Netron
{
    export interface IConnectorTemplate
    {
        name: string;
        type: string;
        description: string;

        getConnectorPosition(element: Element): Point;
    }
}