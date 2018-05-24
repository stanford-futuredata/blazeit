Object detection suffers from poor performance on small objects, so it's
beneficial to:
1. Crop a part of the image.
2. Exclude parts of the image.

We support cropping rectangles and excluding convex polygons. Exclusions are
drawn over in black.

Frames and labels should be jointly processed, ideally using as similar code
as possible.
