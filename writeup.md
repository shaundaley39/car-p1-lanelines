# **Finding Lane Lines on the Road** 

## Writeup

---

## Reflection

### 1. The Lane Line Pipeline

The essential structure of this lane line detector is a 6-stage pipeline into which image frames are fed. The stages respectively are:
- grayscale conversion
- Gaussian noise
- Canny edge detection
- area of interest selection
- Hough transform based line extraction
- fitting a two lane line model to the set of extracted lines

The first five of these steps are implemented using standard OpenCV library operations, in a similar manner to what was explored in previous lessons. The sixth step is implicit, and is included in the draw_lines function. Let's consider each of these steps in a little more detail. As we describe what they do and why, let's also see what they do with this example image:

[image1]: ./test_images/whiteCarLaneSwitch.jpg "Our example image"

Canny finds edges using gradients over the two-dimensional image space - so it's necessary first to reduce three color dimensions to one. We achieve that by converting an input image frame to grayscale. If we ran Canny edge detection directly on this grayscale image we would find many spurious edges - individual pixels can be noisy. We add gaussian noise to the grayscale image as a sort of low pass filter: a larger number of pixels must then show an edge pattern for Canny to find an edge.
[image2]: ./test_images_intermediate/blurred_whiteCarLaneSwitch.jpg "Grayscale with Gaussian noise"

With this pre-processing complete, Canny edge detection is used to find a set of edges on the image:
[image3]: ./test_images_intermediate/edges_whiteCarLaneSwitch.jpg "Canny edges"

We have a particular context in mind: we are searching for lane markings on the road space ahead of a vehicle. We could simplify our search for lanes by focussing on those parts of the image where lanes ought to be: in the lower half, and closer to the centre as lane markings go off into the distance. The area of interest selection implements this idea: it selects only those edges in a quadrilateral area bounding the region where the lane ought to be.
[image4]: ./test_images_intermediate/roi_whiteCarLaneSwitch.jpg "Region of interest"

From a set of edges, we want to infer a set of lines. To do this we make use of Hough transforms: each edge in image space is mapped to a point in Hough Space; a large number of nearby points in Hough Space suggest the existence of a line comprising many of the detected edges.
[image5]: ./test_images_intermediate/hough_lines_whiteCarLaneSwitch.jpg "Hough lines"

After extracting a set of lines from our image, in a region where we would anticipate lane lines, we still have the question: which of these lines are in fact part of the lane lines? And what do the lane lines look like? To deal with this question, we create a simple lane line model (implicit in the draw_lines function). We assume that we will only see (at most) two lane lines: a left line and a right line. We filter each inferred line based on its gradient: negative gradients correspond to the left lane and positive gradients to the right lane (note: y values increase moving down the image, unlike with conventional Cartesian coordinates). For the left and right lane lines respectively, average gradients and offsets are calculated (crude, unweighted). These are then projected from the bottom of the picture towards the centre, and overlaid onto the original image marking inferred lane lines.
[image6]: ./test_images_intermediate/final_whiteCarLaneSwitch.jpg "Lane lines overlaid on original image"


### 2. Likely Shortcomings

There are shortcomings in every stage in this pipeline:
- images are converted to grayscale since this is the standard input to canny edge detection; yet this is throwing away useful information in the search for lanes
- gaussian blur is added (effectively a low pass filter) to reduce noise in the subsequent edge detection, but it is parameterized with a fixed number of pixes. This parameterization is not robust to changing image sizes.
- canny edge detection is implemented here with hard-coded upper and lower gradient thresholds. However real world pixel gradients may vary due to different lighting conditions, different cameras, different calibration or a number of other factors, and pre-set fixed thresholds are not robust to this.
- the region of interest considers only the lower half of the image. While that may be appropriate in 95% of cases (or indeed much more often), it may be a wrong assumption if a car is on going over abrupt hills or undulated terrain.
- the Haugh transform based line extraction depends on a number of parameters, and here there are at least two errors in the approach taken here. Firstly, all but one of the parameters is sensitive to the size of the image, and so the fixed values used are not robust to different sizes of image. Secondly, these parameters were twiddled heavily to find values that worked reasonably well with only 3 test videos and the small number of test images. That is a tiny training set, and the result hasn't been put through testing or validation.
Every stage in the pipeline could be improved along the lines above.

There may be more fundamental shortcomings:
- road lines are often arcs, not straight lines. Our pipeline should support that, but already with the parameterization of the Haugh transform based line detection we probably lost many curves. With our road lane model (finding two straight lane lines to fit observations) we got further from reality for many real-world situations.
- road lines are constrained in other ways: they define road lanes, and the lateral separation distance between them varies within known bounds (depending on jurisdiction). This sort of context could be the basis of a more robust model.
- any model based on gradients alone (Canny) is likely to find spurious lane lines in certain unlucky (but inevitable eventually) lighting/ shadow conditions
- in real world robotics/ self driving cars, the whole pipeline has to execute in milliseconds. Any algorithm or model we use must be parallelized - ideally vectorized to run on GPUs.
- there are many properties of the world that we will need to infer - road lines, road surfaces, drivable surfaces, road signs, vehicles, pedestrians, birds and much else. An ideal pipeline would take a systematic approach to the common aspects of these problems. Manually constructing models for every entity or world-property we want to infer will be laborious; in a sufficiently constrained environment the results might be good enough, whereas in reality they might not be.

### 3. Scope for Improvement

A good moment of reckoning (with additional data points as a bonus) would be to obtain a few additional road videos - perhaps with different camera resolutions, different camera calibrations and moderately different lighting conditions. The present pipeline would fall frustratingly short. That would be an impetus to reconsider. Already, there's a challenge video that fills this function, and it can surely get far worse.

Many parameters depended on image resolution, and those could be scaled based on the area of the image. Canny edge detection thresholds could perhaps be scaled based on some measure of contrast. Ideally we would use scale/ resolution invariant parameters, and at leats make Canny thresholds more robust.

If we could arrive at a set of parameters that were invariant to image resolution, and perhaps less brittle to variation in lighting, we would still need to tune those on a much larger data set - ideally with separate test and validation sets.

A more robust and tightly-constrained parameterized model of lane lines could be developed, which might then be used to reject spurious detected lines and to fit only those detected lines which are likely to be lane markings.

This whole pipeline could be evaluated as to its potential for parallel/ vectorized execution and real time reliability.

Before investing too much there though, it's worth stepping back, considering entirely different approaches (convolutional neural networks? something else) and seeing further results of prior research.
