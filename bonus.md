### 4a.2
For this part, I hypothesize another set of Points of Interests that would better capture the distribution of requests. I used the K-mean clustering algorithm to come up with the new set of coordinates. In addition, because there are many outliers in the bar graph, I used 4 points of interest instead of 3 to cover the outliers resulting from having only 3 POIs

Below are the four POIs from the K-mean algorithm:
+-----+------------------+-------------------+
|POIID|      POI_Latitude|      POI_Longitude|
+-----+------------------+-------------------+
| POI1|51.924950713012386|-113.35158346374011|
| POI2|  43.9139049192433| -79.95142058260134|
| POI3| 25.13193480000001|  72.13771439999998|
| POI4| 45.67683093901952| -71.91334253273482|
+-----+------------------+-------------------+

As a comparison, I listed the mean, standard deviation, radius, and request density for the new and old POIs:

Average distance for each NEW Point of Interest
+-----+------------------+
|POIID|     avg(Distance)|
+-----+------------------+
| POI4|291.57883345611026|
| POI2|105.60132893576147|
| POI1| 259.4796987203471|
| POI3| 3465.434658571387|
+-----+------------------+

The radius of circle enclosing all requests for each NEW Point of Interest
+-----+------------------+
|POIID|     max(Distance)|
+-----+------------------+
| POI4| 5839.317627684705|
| POI2|  1306.22950139066|
| POI1|1826.1560249033323|
| POI3| 6008.679788738752|
+-----+------------------+

Standard deviation of distance for each NEW Point of Interest
+-----+---------------------+
|POIID|stddev_samp(Distance)|
+-----+---------------------+
| POI4|    302.4213149490086|
| POI2|    162.5148195922189|
| POI1|   254.73102161834493|
| POI3|   1501.5031572626394|
+-----+---------------------+

Request density for each NEW Point of Interest
+-----+-----+------------------+--------------------+
|POIID|Count|            Radius|             Density|
+-----+-----+------------------+--------------------+
| POI4| 2163| 5839.317627684705|2.019214463313605...|
| POI2| 7076|  1306.22950139066|0.001320076121400...|
| POI1| 8717|1826.1560249033323|8.320346491454754E-4|
| POI3|   17| 6008.679788738752|1.498790486578416E-7|
+-----+-----+------------------+--------------------+

Average distance for each Point of Interest
+-----+------------------+
|POIID|     avg(Distance)|
+-----+------------------+
| POI4| 514.9971719812286|
| POI1|300.71474756868054|
| POI3| 451.6511492015015|
+-----+------------------+

The radius of circle enclosing all requests for each Point of Interest
+-----+------------------+
|POIID|     max(Distance)|
+-----+------------------+
| POI4| 9349.572770487366|
| POI1|11531.820831836454|
| POI3|1474.5809620285709|
+-----+------------------+

Standard deviation of distance for each Point of Interest
+-----+---------------------+
|POIID|stddev_samp(Distance)|
+-----+---------------------+
| POI4|   1506.8899707703208|
| POI1|   388.27338526354424|
| POI3|   223.63174183104917|
+-----+---------------------+

Request density for each Point of Interest
+-----+-----+------------------+--------------------+
|POIID|Count|            Radius|             Density|
+-----+-----+------------------+--------------------+
| POI4|  422| 9349.572770487366|1.536664455904176...|
| POI1| 8749|11531.820831836454|2.094174038984837...|
| POI3| 8802|1474.5809620285709|0.001288529145748...|
+-----+-----+------------------+--------------------+

In general, the distance between the new POIs and the requests is shorter on average, the standard deviation is smaller, and the enclosing circle's radius is also smaller. This means that the requests reside more closely around the new POIs than the old ones.