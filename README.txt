Integration GroIMP / OpenAlea:

To communicate between GroIMP and OpenAlea, two solutions appears.
The first solution is to communicate from OpenAlea to GroIMP through 
an http connection.

1. GroIMP is started as the server, 
2. OpenAlea send the XL file to GroIMP + a graph if needed.
3. GroIMP start a project and compile the file
4. OA send commands to GroIMP for eecution
5. GroIMP returns the results as an XML file
6. OA parse the file and construct internal data structure:
  * scenegraph and MTG/Graph

Geometric Objects:
Box
Sphere
Cylinder
Cone
Transform


TriangleSet
FaceSet
ElevationGrid
Frustum
NurbsPatch

