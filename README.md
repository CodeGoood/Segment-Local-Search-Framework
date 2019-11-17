# Segment Local Search Framework
We will introduce the proposed framework, segmented local search.

## Segmented Local Search Framework
In TSP, the running time of many local search algorithms increases as the number of cities
grows. Therefore, the large scale TSP instance takes a long time to obtain good solution
quality. Therefore, we want to address the issue in terms of the neighborhood size of local
search. In order to achieve the goal, we propose a segmented local search framework to
narrow down the neighborhood size of local search. The main idea is to optimize the
fragments of tour instead of the whole tour. In this study, we adopt best 2-opt and LKH
as the local search in the proposed framework, which is seg2opt, and segLKH respectively.
This framework can significantly increase the convergence speed and also maintain the
solution quality.

### Tour Segmentation
A tour ***T*** will be randomly cut into several segments, which is also called sub-tours. 
Each segment has the same segment length ***l***, which is equal to 4 in the beginning. Following picture shows the partiition of a tour.  
<img src="https://github.com/CodeGoood/Segment-Local-Search-Framework/blob/master/pic/Tour%20segmentation.png" width="500" height="270">  

Given ***N*** cities, the number of total segments ***s*** is set to ⌈***N/l***⌉. 
Every segment can be conducted the local search, and will not produce illegal route.
Therefore, this framework is a single local search parallelization. At each iteration, when
all segments finish their local search once, the cutting points of the tour will be randomly
shifted to avoid the local optimal. We will increase the segment length ***l*** when a certain number of segments get
convergence. Finally, the segmented local search framework
stops when segment length ***l*** equals to ***N*** , and a better move for the whole tour cannot be
found. Next, we will introduce the adaptive segment length ***l*** mechanism.

### Adaptive Control of Segment Length
When all the segments executed local search once, variable ***p*** denotes the number of nonimproved
segments, where ***p***<=***s***. The adaptive control of segment length is the following
formula, and the ***i***<sup>th</sup> segment length ***l***<sub>i</sub> is determined by the following fomula:  
<img src="https://github.com/CodeGoood/Segment-Local-Search-Framework/blob/master/pic/form.png" width="200" height="100">

The value ***p/s*** denotes the degree of convergence of the whole tour, and parameter ***θ***
denotes the convergence threshold, where ***θ*** ∈ (0,1]. If ***p/s*** is greater than the convergence
threshold ***θ***, which means that plenty of segments have become convergence. At this time,
the segment length ***l*** will be multiplied the parameter ***η*** (growing factor), where η ∈ ℝ<sup>+</sup>.
If ***p/s*** is smaller than the convergence threshold ***θ***, the segment length remains the same
length. By doing so, the search can jump away from local optimal, so that the local search
can find a better moves. Also, the mechanism can let the local search obtain a better convergence
speed. When segment length ***l*** is greater than ***N***, it equals to ***N***. The framework
terminates when segment length ***l*** equals ***N*** and the whole tour cannot be improved.  

#### The following algorithm depicts the pseudocode of segmented local search framework.
<img src="https://github.com/CodeGoood/Segment-Local-Search-Framework/blob/master/pic/alo.png" width="500" height="450">
