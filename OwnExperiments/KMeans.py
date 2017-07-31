



def rotate90degrees(c1, c2):
    (x1, y1) = c1
    (x2, y2) = c2

    xMid = float(x1 + x2)/2
    yMid = float(y1 + y2)/2

    xDelta = abs(x2-x1)
    xDeltaHalf = xDelta/2
    yDelta = abs(y2-y1)
    yDeltaHalf = yDelta/2

    return ((xMid-yDeltaHalf,yMid+xDeltaHalf),(xMid+yDeltaHalf,yMid-xDeltaHalf))


print rotate90degrees((6,5),(8,10))
