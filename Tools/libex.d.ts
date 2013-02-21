
interface TouchEvent extends UIEvent {
    touches: TouchList;
    targetTouches: TouchList;
    changedTouches: TouchList;
    altKey: bool;
    metaKey: bool;
    ctrlKey: bool;
    shiftKey: bool;
    relatedTarget: EventTarget;
}

interface TouchList {
    length: number;
    item(index: number): Touch;
    [index: number]: Touch;
    identifiedTouch(identifier: number): Touch;
}

interface Touch {
    identifier: number;
    target: EventTarget;
    screenX: number;
    screenY: number;
    clientX: number;
    clientY: number;
    pageX: number;
    pageY: number;
    radiusX: number;
    radiusY: number;
    rotationAngle: number;
    force: number;
}
