# TODO

# 1. Implement the full specification
# 2. Test all the cases with several examples
# 3. Implement a loop in OpenAlea
# 4. Use the PlantGL turtle

# 2.1. add a header
# 2.2. separate graph parsing and scenegraph generation
# 2.3. Error management
# 2.4. Documentation
# 2.5. Compute properties when it is possible (sphere, ...)
# 2.6. 2D draw of the graph


# 3. Add enum like FUNCTIONAL

from math import radians
from math import sqrt
from math import cos
from math import sin
from copy import deepcopy

import openalea.plantgl.all as pgl

Vector3 = pgl.Vector3
Vector4 = pgl.Vector4
Color4Array = pgl.Color4Array


def rgb_color(index):
    """ Returns an RGB color from an index. """
    LUT_color = [(0, 0, 0),
                 (0, 0, 170),
                 (0, 170, 0),
                 (0, 170, 170),
                 (170, 0, 0),
                 (170, 0, 170),
                 (170, 85, 0),
                 (170, 170, 170),
                 (85, 85, 85),
                 (85, 85, 255),
                 (85, 255, 85),
                 (85, 255, 255),
                 (255, 85, 85),
                 (255, 85, 255),
                 (255, 255, 85),
                 (255, 255, 255), ]
    return pgl.Color3(LUT_color[int(index)])


class TurtleState(object):
    """ Store the turtle state of each vertex. """
    DIAMETER = 0.1
    LENGTH = 0.1

    def __init__(self):
        self.diameter = -1
        self.diameter_add = 0
        self.diameter_mul = 1.
        self.set_diameter = False
        self.color = None
        self.length = -1.
        self.length_add = 0
        self.length_mul = 1.
        self.set_length = False

        self.tropism = 0.
        self.tropism_direction = None
        self.tropism_target = None

    def copy(self):
        return deepcopy(self)

    def combine(self, t):
        # copy to avoid side-effect
        cself = self.copy()
        if t.diameter != -1:
            cself.diameter = t.diameter
        if t.diameter_add != 0:
            if cself.diameter == -1:
                cself.diameter = self.DIAMETER
            cself.diameter += t.diameter_add
        if t.diameter_mul != 1:
            if cself.diameter == -1:
                cself.diameter = self.DIAMETER
            cself.diameter *= t.diameter_mul
        if t.set_diameter:
            if cself.diameter == -1 and t.diameter == -1:
                cself.diameter = self.DIAMETER
            elif cself.diameter == -1:
                cself.diameter = t.diameter

        self.color = t.color

        if t.length != -1:
            cself.length = t.length
        if t.length_add != 0:
            if cself.length == -1:
                cself.length = self.LENGTH
            cself.length += t.length_add
        if t.length_mul != 1:
            if cself.length == -1:
                cself.length = self.LENGTH
            cself.length *= t.length_mul
        if t.set_length:
            if cself.length == -1 and t.length == -1:
                cself.length = self.DIAMETER
            elif cself.length == -1:
                cself.length = t.length

        cself.tropism = t.tropism
        cself.tropism_direction = t.tropism_direction
        cself.tropism_target = t.tropism_target

        return cself
    def __eq__(self, other):
        """ Test for == operator """
        ok = ((self.diameter == other.diameter) and
              (self.diameter_add == other.diameter_add) and
              (self.diameter_mul == other.diameter_mul) and
              (self.set_diameter == other.set_diameter) and
              (self.color == other.color) and
              (self.length == other.length) and
              (self.length_add == other.length_add) and
              (self.length_mul == other.length_mul) and
              (self.set_length == other.set_length) and
              (self.tropism == other.tropism) and
              (self.tropism_direction == other.tropism_direction) and
              (self.tropism_target == other.tropism_target)
              )
        return ok

    def __nonzero__(self):
        return not self.__eq__(TurtleState())


class FunctionalGeometry(object):
    def __init__(self, function):
        self.function = function

    def __call__(self, turtle_state):
        return self.function(turtle_state)


#######################################


def is_matrix(shape):
    return type(shape) == pgl.Matrix4


def transform4(matrix, shape):
    """
    Return a shape transformed by a Matrix4.
    """
    scale, (a, e, r), translation = matrix.getTransformation2()
    shape = pgl.Translated(translation,
                           pgl.Scaled(scale,
                                      pgl.EulerRotated(a, e, r,
                                                       shape)))
    return shape


##########################################################################
# Debug utility

def frame(matrix, scene, color=1):
    """ Add a frame to the scene.
    The frame is represented by the matrix.
    :param color: allow to distinguish between to identical frames.
    """
    if color == 1:
        r = pgl.Material(pgl.Color3(*(255, 0, 0)))
        g = pgl.Material(pgl.Color3(*(0, 255, 0)))
        b = pgl.Material(pgl.Color3(*(0, 0, 255)))
    else:
        r = pgl.Material(pgl.Color3(*(255, 0, 255)))
        g = pgl.Material(pgl.Color3(*(255, 255, 0)))
        b = pgl.Material(pgl.Color3(*(0, 0, 0)))

    cylinder = pgl.Cylinder(radius=0.005, height=1)
    #x = pgl.AxisRotated(Vector3(0,1,0), radians(90), cylinder)
    #y = pgl.AxisRotated(Vector3(1,0,0), radians(-90), cylinder)
    z = cylinder

    #geom_x = transform4(matrix, x)
    #scene.add(pgl.Shape(geom_x, r))
    #geom_y = transform4(matrix, y)
    #scene.add(pgl.Shape(geom_y, g))
    geom_z = transform4(matrix, z)
    scene.add(pgl.Shape(geom_z, b))

##########################################################################
